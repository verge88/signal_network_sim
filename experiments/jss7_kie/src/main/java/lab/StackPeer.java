package lab;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.mobicents.protocols.api.IpChannelType;
import org.restcomm.protocols.ss7.map.load.Client;
import org.restcomm.protocols.ss7.map.load.Server;
import org.restcomm.protocols.ss7.map.api.service.supplementary.ProcessUnstructuredSSRequest;
import org.restcomm.protocols.ss7.map.api.service.supplementary.ProcessUnstructuredSSResponse;
import org.restcomm.protocols.ss7.m3ua.impl.M3UAManagementImpl;
import org.restcomm.protocols.ss7.m3ua.As;
import org.restcomm.protocols.ss7.m3ua.Functionality;
import org.restcomm.protocols.ss7.m3ua.ExchangeType;
import org.restcomm.protocols.ss7.m3ua.IPSPType;
import org.restcomm.protocols.ss7.m3ua.parameter.TrafficModeType;
import org.restcomm.protocols.ss7.m3ua.impl.parameter.ParameterFactoryImpl;
import org.mobicents.protocols.sctp.netty.NettySctpManagementImpl;
import com.google.common.util.concurrent.RateLimiter;

/** Small adapter around the pinned upstream MAP load peers.
 * Uses real MAP/TCAP/SCCP/M3UA/SCTP, with no TCP fallback.
 * Reflection is confined to the upstream example's private setup/send API.
 */
public final class StackPeer {
    private static void setField(Object peer, Class<?> base, String name, Object value) throws Exception {
        Field field = base.getDeclaredField(name);
        field.setAccessible(true);
        field.set(peer, value);
    }
    private static void waitActive(M3UAManagementImpl management) throws Exception {
        Method getAs = M3UAManagementImpl.class.getDeclaredMethod("getAs", String.class);
        getAs.setAccessible(true);
        As applicationServer = (As) getAs.invoke(management, "AS1");
        long deadline = System.nanoTime() + 45_000_000_000L;
        while (!"ACTIVE".equalsIgnoreCase(applicationServer.getState().getName())) {
            if (System.nanoTime() > deadline) throw new IllegalStateException("M3UA AS did not become ACTIVE");
            Thread.sleep(100);
        }
    }
    // Both MAP endpoints are ASPs when an external SGP routes between them.
    private static M3UAManagementImpl initializeViaStp(Object peer, Class<?> base,
            String fieldName, int localPort, long routingContext, int remotePointCode) throws Exception {
        NettySctpManagementImpl transport = new NettySctpManagementImpl("StpPeer" + localPort);
        transport.start();
        transport.setConnectDelay(1000);
        transport.removeAllResourses();
        transport.addAssociation("127.0.0.1", localPort, "127.0.0.1", 2905, "stp", IpChannelType.SCTP, null);
        setField(peer, base, "sctpManagement", transport);
        M3UAManagementImpl management = new M3UAManagementImpl("StpM3ua" + localPort, null, null);
        management.setTransportManagement(transport);
        management.setDeliveryMessageThreadCount(1);
        management.start();
        management.removeAllResourses();
        ParameterFactoryImpl parameters = new ParameterFactoryImpl();
        management.createAs("AS1", Functionality.AS, ExchangeType.SE, IPSPType.CLIENT,
            parameters.createRoutingContext(new long[] {routingContext}),
            parameters.createTrafficModeType(TrafficModeType.Loadshare), 1, null);
        management.createAspFactory("ASP1", "stp");
        management.assignAspToAs("AS1", "ASP1");
        management.addRoute(remotePointCode, -1, -1, "AS1");
        setField(peer, base, fieldName, management);
        for (String name : new String[] {"initSCCP", "initTCAP", "initMAP"}) {
            Method method = base.getDeclaredMethod(name);
            method.setAccessible(true);
            method.invoke(peer);
        }
        management.startAsp("ASP1");
        return management;
    }
    private static final class Receiver extends Server {
        private final Path events;
        private final Set<Long> seen = new HashSet<Long>();
        Receiver(Path path) { events = path; }
        void start(boolean stp) throws Exception {
            CLIENT_IP = SERVER_IP = "127.0.0.1";
            CLIENT_PORT = 2345; SERVER_PORT = 3434;
            DELIVERY_TRANSFER_MESSAGE_THREAD_COUNT = 1;
            if (stp) waitActive(initializeViaStp(this, Server.class, "serverM3UAMgmt", 3434, 102, CLIENT_SPC));
            else initializeStack(IpChannelType.SCTP);
        }
        @Override public synchronized void onProcessUnstructuredSSRequest(ProcessUnstructuredSSRequest request) {
            long id = request.getMAPDialog().getLocalDialogId();
            if (seen.add(id)) {
                try {
                    Files.write(events, (id + "\n").getBytes(StandardCharsets.UTF_8),
                        StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                } catch (Exception e) { e.printStackTrace(); System.exit(2); }
            }
            super.onProcessUnstructuredSSRequest(request);
        }
    }
    private static final class Sender extends Client {
        private final AtomicInteger completed = new AtomicInteger();
        private final Set<Long> seen = new HashSet<Long>();
        @Override public synchronized void onProcessUnstructuredSSResponse(ProcessUnstructuredSSResponse response) {
            if (seen.add(response.getMAPDialog().getLocalDialogId())) completed.incrementAndGet();
        }
        void run(int count, boolean stp, int pacingMs, Path stopFile) throws Exception {
            CLIENT_IP = SERVER_IP = "127.0.0.1";
            CLIENT_PORT = 2345; SERVER_PORT = 3434;
            DELIVERY_TRANSFER_MESSAGE_THREAD_COUNT = 1;
            MAXCONCURRENTDIALOGS = 20;
            if (stp) {
                setField(this, Client.class, "rateLimiterObj", RateLimiter.create(MAXCONCURRENTDIALOGS));
                // The upstream private send method also increments this CSV counter.
                Class<?> writerType = Class.forName("org.restcomm.protocols.ss7.map.load.CsvWriter");
                java.lang.reflect.Constructor<?> constructor = writerType.getDeclaredConstructor(String.class);
                constructor.setAccessible(true);
                Object writer = constructor.newInstance("map");
                Method add = writerType.getDeclaredMethod("addCounter", String.class);
                add.setAccessible(true);
                for (String counter : new String[] {CREATED_DIALOGS, SUCCESSFUL_DIALOGS, ERROR_DIALOGS}) add.invoke(writer, counter);
                setField(this, Client.class, "csvWriter", writer);
                initializeViaStp(this, Client.class, "clientM3UAMgmt", 2345, 101, SERVET_SPC);
            } else initializeStack(IpChannelType.SCTP);
            Field field = Client.class.getDeclaredField("clientM3UAMgmt");
            field.setAccessible(true);
            M3UAManagementImpl management = (M3UAManagementImpl) field.get(this);
            waitActive(management);
            Method send = Client.class.getDeclaredMethod("initiateUSSD");
            send.setAccessible(true);
            System.out.println("TRAFFIC_READY");
            for (int i = 1; i <= count; i++) {
                if (stopFile != null && Files.exists(stopFile)) break;
                send.invoke(this);
                long deadline = System.nanoTime() + 10_000_000_000L;
                while (completed.get() < i) {
                    if (System.nanoTime() > deadline) throw new IllegalStateException("MAP response timeout at " + i);
                    Thread.sleep(10);
                }
                if (pacingMs > 0) Thread.sleep(pacingMs);
            }
            System.out.println("COMPLETED=" + completed.get());
        }
    }
    public static void main(String[] args) {
        try {
            if ((args.length != 3 && args.length != 5) || !("direct".equals(args[2]) || "osmostp".equals(args[2])))
                throw new IllegalArgumentException("server EVENT_FILE TOPOLOGY | client COUNT TOPOLOGY [PACING_MS STOP_FILE]");
            boolean stp = "osmostp".equals(args[2]);
            if ("server".equals(args[0])) {
                Receiver server = new Receiver(Paths.get(args[1]));
                server.start(stp);
                System.out.println("SERVER_READY");
                Thread.currentThread().join();
            } else if ("client".equals(args[0])) {
                int count = Integer.parseInt(args[1]);
                if (count < 1 || count > 1000) throw new IllegalArgumentException("COUNT must be 1..1000");
                int pacing = args.length == 5 ? Integer.parseInt(args[3]) : 0;
                if (pacing < 0 || pacing > 1000) throw new IllegalArgumentException("invalid pacing");
                new Sender().run(count, stp, pacing, args.length == 5 ? Paths.get(args[4]) : null);
                System.exit(0);
            } else throw new IllegalArgumentException("unknown role");
        } catch (Exception e) { e.printStackTrace(); System.exit(2); }
    }
}
