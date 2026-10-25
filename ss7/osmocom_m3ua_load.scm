;;; Controlled M3UA DATA generator for SS7 transport calibration.
;;; Loaded inside the nplab/m3ua-testtool container after the official
;;; conformance suite has completed.  /root/.guile already loads m3ua.scm
;;; and /data/m3ua-param-testtool.scm.

(define calibration-si 15)
(define calibration-payload-sizes '(24 40 56 72 92 152 192 256))
(define calibration-frames-per-size 30)
(define calibration-gap-us 20000)

(define (send-controlled-data fd payload-size seq)
  (m3ua-send-message
   fd
   1
   (m3ua-make-data-message
    tester-pc
    iut-pc
    calibration-si
    iut-ni
    iut-mp
    (modulo seq 16)
    (zero-bytes payload-size)
    data-message-parameters)))

(define (run-controlled-load)
  (let ((fd (m3ua-connect tester-addr tester-port sut-addr sut-port)))
    (m3ua-send-message fd 0
      (m3ua-make-asp-up-message asp-up-message-parameters))
    (m3ua-wait-for-message fd m3ua-asp-up-ack-message?)
    (m3ua-send-message fd 0
      (m3ua-make-asp-active-message asp-active-message-parameters))
    (m3ua-wait-for-message fd m3ua-asp-active-ack-message?)
    (sleep 1)

    (let ((seq 0))
      (for-each
       (lambda (payload-size)
         (format #t "CALIBRATION_PAYLOAD_BEGIN,size=~a,count=~a~%"
                 payload-size calibration-frames-per-size)
         (dotimes (i calibration-frames-per-size)
           (send-controlled-data fd payload-size seq)
           (set! seq (+ seq 1))
           (usleep calibration-gap-us))
         (format #t "CALIBRATION_PAYLOAD_END,size=~a~%" payload-size))
       calibration-payload-sizes))

    (sleep 1)
    (close fd)))

(format #t "CALIBRATION_LOAD_START~%")
(run-controlled-load)
(format #t "CALIBRATION_LOAD_END~%")
