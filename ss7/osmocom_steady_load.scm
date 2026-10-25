;;; Controlled M3UA DATA workload for SS7 model calibration.
;;; Loaded inside the official nplab/m3ua-testtool container after the
;;; conformance suite. This creates a separate, steady stochastic workload;
;;; it is not operator traffic.

(load "/root/.guile")

(define rate-hz 60.0)
(define message-count 1800)
(define min-sleep-us 100)

(define (uniform-open)
  ;; Guile 2.0 random accepts an integer upper bound. Keep the result in (0,1).
  (/ (+ 1.0 (random 1000000)) 1000001.0))

(define (poisson-wait-us lambda)
  (let* ((u (uniform-open))
         (seconds (/ (- (log u)) lambda))
         (usec (inexact->exact (round (* seconds 1000000.0)))))
    (max min-sleep-us usec)))

(define fd (m3ua-connect tester-addr 0 sut-addr sut-port))

(m3ua-send-message fd 0
                   (m3ua-make-asp-up-message asp-up-message-parameters))
(m3ua-wait-for-message fd m3ua-asp-up-ack-message?)

(m3ua-send-message fd 0
                   (m3ua-make-asp-active-message asp-active-message-parameters))
(m3ua-wait-for-message fd m3ua-asp-active-ack-message?)

(display "steady-load-start rate_hz=")
(display rate-hz)
(display " messages=")
(display message-count)
(newline)

(dotimes (n message-count)
  (m3ua-send-message
    fd
    1
    (m3ua-make-data-message
      opc dpc si ni mp sls ss7-message data-message-parameters))
  (usleep (poisson-wait-us rate-hz)))

(sleep 1)
(close fd)

(display "steady-load-finished")
(newline)
