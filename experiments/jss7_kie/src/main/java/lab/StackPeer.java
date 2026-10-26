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

/** Small adapter around the pinned upstream MAP load peers.
 * Uses real MAP/TCAP/SCCP/M3UA/SCTP, with no TCP fallback.
 * Reflection is confined to the upstream example's private setup/send API.
 */
public final class StackPeer {
    private static final class Receiver extends Server {
        private final Path events;
        private final Set<Long> seen = new HashSet<Long>();
        Receiver(Path path) { events = path; }
        void start() throws Exception {
            CLIENT_IP = SERVER_IP = "127.0.0.1";
            CLIENT_PORT = 2345; SERVER_PORT = 3434;
            DELIVERY_TRANSFER_MESSAGE_THREAD_COUNT = 1;
            initializeStack(IpChannelType.SCTP);
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
        void run(int count) throws Exception {
            CLIENT_IP = SERVER_IP = "127.0.0.1";
            CLIENT_PORT = 2345; SERVER_PORT = 3434;
            DELIVERY_TRANSFER_MESSAGE_THREAD_COUNT = 1;
            MAXCONCURRENTDIALOGS = 20;
            initializeStack(IpChannelType.SCTP);
            Field field = Client.class.getDeclaredField("clientM3UAMgmt");
            field.setAccessible(true);
            M3UAManagementImpl management = (M3UAManagementImpl) field.get(this);
            Method getAs = M3UAManagementImpl.class.getDeclaredMethod("getAs", String.class);
            getAs.setAccessible(true);
            As applicationServer = (As) getAs.invoke(management, "AS1");
            long deadline = System.nanoTime() + 45_000_000_000L;
            while (!"ACTIVE".equalsIgnoreCase(applicationServer.getState().getName())) {
                if (System.nanoTime() > deadline) throw new IllegalStateException("M3UA AS did not become ACTIVE");
                Thread.sleep(100);
            }
            Method send = Client.class.getDeclaredMethod("initiateUSSD");
            send.setAccessible(true);
            for (int i = 1; i <= count; i++) {
                send.invoke(this);
                deadline = System.nanoTime() + 10_000_000_000L;
                while (completed.get() < i) {
                    if (System.nanoTime() > deadline) throw new IllegalStateException("MAP response timeout at " + i);
                    Thread.sleep(10);
                }
            }
            System.out.println("COMPLETED=" + completed.get());
        }
    }
    public static void main(String[] args) {
        try {
            if (args.length != 2) throw new IllegalArgumentException("server EVENT_FILE | client COUNT");
            if ("server".equals(args[0])) {
                Receiver server = new Receiver(Paths.get(args[1]));
                server.start();
                System.out.println("SERVER_READY");
                Thread.currentThread().join();
            } else if ("client".equals(args[0])) {
                int count = Integer.parseInt(args[1]);
                if (count < 1 || count > 1000) throw new IllegalArgumentException("COUNT must be 1..1000");
                new Sender().run(count);
                System.exit(0);
            } else throw new IllegalArgumentException("unknown role");
        } catch (Exception e) { e.printStackTrace(); System.exit(2); }
    }
}
