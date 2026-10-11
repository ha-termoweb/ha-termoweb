/* termoweb_rx for the nanoCUL868: Arduino Nano (ATmega328P, 16 MHz) with a
 * CC1101 on SPI, talking the termoweb_rx line protocol at 115200 8N1.
 *
 * This file is only the AVR platform layer: UART, SPI, timers, the two GDO
 * interrupts and the raw-survey edge capture. Everything the ESP32 radio
 * shares (dialects, acks, command parser, replies, receive state machine)
 * is in ../radio_core.
 *
 * Wiring (a-culfw Devices/nanoCUL/board.h):
 *   MOSI PB3/D11, MISO PB4/D12, SCK PB5/D13, CS PB2/D10 (manual GPIO),
 *   GDO0 PD3/D3 (INT1: packet done; raw demodulator output in a survey),
 *   GDO2 PD2/D2 (INT0: RX FIFO at threshold).
 *
 * Opening the serial port pulses DTR, which resets the Nano: every new
 * host connection starts from power-on defaults and the banner, exactly
 * like a new TCP connection to the ESP32.
 */
#include <avr/interrupt.h>
#include <avr/io.h>
#include <util/delay.h>
#include <util/delay_basic.h>

#include "cc1101.h"
#include "radio_core.h"
#include "rc_hal.h"

/* ------------------------------------------------------------ clock */

static volatile uint32_t timer0_ovf;

ISR(TIMER0_OVF_vect) { timer0_ovf++; }

/* Arduino-style micros(): timer0 at F_CPU/64 = 4 us per tick. */
uint32_t rc_hal_micros(void) {
  uint8_t sreg = SREG;
  cli();
  uint32_t ovf = timer0_ovf;
  uint8_t cnt = TCNT0;
  if ((TIFR0 & (1 << TOV0)) && cnt < 255) ovf++;
  SREG = sreg;
  return ((ovf << 8) | cnt) * 4UL;
}

static void timer0_init(void) {
  TCCR0A = 0;
  TCCR0B = (1 << CS01) | (1 << CS00); /* prescaler 64 */
  TIMSK0 = (1 << TOIE0);
}

/* _delay_loop_2: 4 cycles per count. */
void rc_hal_delay_us(uint16_t us) {
  while (us > 4000) {
    _delay_loop_2((uint16_t)(4000UL * (F_CPU / 4000000UL)));
    us -= 4000;
  }
  if (us) _delay_loop_2((uint16_t)(us * (F_CPU / 4000000UL)));
}

/* ------------------------------------------------------------- UART */

/* 256-byte ring so the index wraps by itself; room for a 99-byte program
 * write's 'T' line (199 characters) arriving while a long RX line prints. */
static volatile uint8_t uart_rx_buf[256];
static volatile uint8_t uart_rx_head;
static volatile uint8_t uart_rx_tail;
static volatile uint8_t uart_rx_overrun;

ISR(USART_RX_vect) {
  uint8_t c = UDR0;
  uint8_t head = uart_rx_head;
  uint8_t next = (uint8_t)(head + 1);
  if (next == uart_rx_tail) {
    uart_rx_overrun = 1; /* drop the new byte, report once */
    return;
  }
  uart_rx_buf[head] = c;
  uart_rx_head = next;
}

static void uart_init(void) {
  /* 115200 at 16 MHz with U2X0: UBRR 16 (117647 actual, +2.1%). */
  UBRR0H = 0;
  UBRR0L = 16;
  UCSR0A = (1 << U2X0);
  UCSR0B = (1 << RXEN0) | (1 << TXEN0); /* RXCIE0 is set in main(), with sei() */
  UCSR0C = (1 << UCSZ01) | (1 << UCSZ00);
}

static uint8_t uart_getc(char *c) {
  uint8_t tail = uart_rx_tail;
  if (tail == uart_rx_head) return 0;
  *c = (char)uart_rx_buf[tail];
  uart_rx_tail = (uint8_t)(tail + 1);
  return 1;
}

void rc_hal_putc(char c) {
  while (!(UCSR0A & (1 << UDRE0))) {
  }
  UDR0 = (uint8_t)c;
}

/* -------------------------------------------------------------- SPI */

#define CS_BIT PB2
#define MISO_BIT PB4

/* CC1101 datasheet 10.1: after CS low, wait for MISO low (crystal running).
 * Bounded (~8 ms), so a missing module cannot hang the firmware. */
static void cs_low(void) {
  PORTB &= (uint8_t) ~(1 << CS_BIT);
  for (uint16_t n = 0; (PINB & (1 << MISO_BIT)) && n < 20000; n++) {
  }
}

static void cs_high(void) { PORTB |= (1 << CS_BIT); }

static uint8_t spi(uint8_t v) {
  SPDR = v;
  while (!(SPSR & (1 << SPIF))) {
  }
  return SPDR;
}

static void spi_init(void) {
  DDRB |= (1 << PB3) | (1 << PB5) | (1 << CS_BIT); /* MOSI, SCK, CS out */
  DDRB &= (uint8_t) ~(1 << MISO_BIT);
  cs_high();
  SPCR = (1 << SPE) | (1 << MSTR); /* mode 0, MSB first, F_CPU/4 = 4 MHz */
  SPSR = 0;
}

/* Every register access is atomic: the GDO interrupts do SPI themselves,
 * so a main-loop transaction must never be cut in half by one. */
#define SPI_BEGIN()    \
  uint8_t sreg = SREG; \
  cli();               \
  cs_low()
#define SPI_END() \
  cs_high();      \
  SREG = sreg

void rc_hal_chip_wake(void) {
  SPI_BEGIN();
  SPI_END();
  _delay_us(40);
}

uint8_t rc_hal_strobe(uint8_t cmd) {
  SPI_BEGIN();
  uint8_t status = spi(cmd);
  SPI_END();
  return status;
}

void rc_hal_write_reg(uint8_t addr, uint8_t val) {
  SPI_BEGIN();
  spi(addr);
  spi(val);
  SPI_END();
}

void rc_hal_write_burst(uint8_t addr, const uint8_t *buf, uint8_t len) {
  SPI_BEGIN();
  spi(addr | CC1101_WRITE_BURST);
  for (uint8_t i = 0; i < len; i++) spi(buf[i]);
  SPI_END();
}

uint8_t rc_hal_read_status(uint8_t addr) {
  SPI_BEGIN();
  spi(addr | CC1101_READ_BURST);
  uint8_t v = spi(0);
  SPI_END();
  return v;
}

void rc_hal_read_burst(uint8_t addr, uint8_t *buf, uint8_t len) {
  SPI_BEGIN();
  spi(addr | CC1101_READ_BURST);
  for (uint8_t i = 0; i < len; i++) buf[i] = spi(0);
  SPI_END();
}

/* ------------------------------------------------------- GDO edges */

static void gdo_init(void) {
  DDRD &= (uint8_t) ~((1 << PD3) | (1 << PD2));
  EIMSK = 0;
  /* INT1 (GDO0) falling edge = packet done; INT0 (GDO2) rising = FIFO threshold. */
  EICRA = (1 << ISC11) | (1 << ISC01) | (1 << ISC00);
}

void rc_hal_gdo_mask(void) { EIMSK &= (uint8_t) ~((1 << INT0) | (1 << INT1)); }

void rc_hal_gdo_unmask(void) {
  EIFR = (1 << INTF0) | (1 << INTF1); /* forget edges our own transmit caused */
  EIMSK |= rc.survey_active ? (1 << INT1) : ((1 << INT0) | (1 << INT1));
}

void rc_hal_lock(void) { cli(); }
void rc_hal_unlock(void) { sei(); }

ISR(INT0_vect) { rc_on_gdo2(); }

/* ----------------------------------------------------------- survey */

/* Raw survey: the CC1101 in asynchronous serial mode puts the raw 2-FSK
 * demodulator output on GDO0; every edge stores the run that just ended as
 * level << 15 | microseconds (timer1 at 0.5 us per tick, runs of 32 ms and
 * more saturate at 32767). The runs ring overlays the packet buffers
 * (rc.mem.runs, 256 entries = 512 bytes), so one burst keeps up to 48 runs
 * from before the RSSI trigger and up to 208 after it: the preamble, sync
 * word and roughly the first 20 bytes of a frame. */
#define SV_RING 256
#define SV_PRE_RUNS 48
#define SV_POST_US 100000UL
#define SV_RUNS_PER_LINE 60

static volatile uint8_t sv_capture;
static volatile uint16_t sv_head; /* runs written, free-running */
static volatile uint16_t sv_last_t;
static volatile uint8_t sv_last_level;
static volatile uint8_t sv_ovf;
static uint32_t sv_end_us, sv_trig_us;
static uint16_t sv_trig_head, sv_dump_pos, sv_dump_end, sv_dump_runs, sv_bursts;
static uint8_t sv_triggered, sv_dumping;
static int16_t sv_thr, sv_peak;

ISR(TIMER1_OVF_vect) {
  if (sv_ovf < 2) sv_ovf++;
}

static void survey_edge(void) {
  uint16_t t = TCNT1;
  uint8_t ovf = sv_ovf;
  if (TIFR1 & (1 << TOV1)) { /* an overflow not counted yet */
    TIFR1 = (1 << TOV1);
    t = TCNT1;
    ovf++;
  }
  if (sv_capture) {
    uint16_t us = (ovf >= 2 || (ovf == 1 && t >= sv_last_t)) ? 0x7FFF : (uint16_t)(t - sv_last_t) >> 1;
    rc.mem.runs[(uint8_t)sv_head] = (uint16_t)((uint16_t)sv_last_level << 15) | us;
    sv_head++;
  }
  sv_last_t = t;
  sv_last_level = (PIND >> PD3) & 1;
  sv_ovf = 0;
}

ISR(INT1_vect) {
  if (rc.survey_active) {
    survey_edge();
    return;
  }
  rc_on_gdo0(rc_hal_micros());
}

static int16_t rssi_dbm10(void) { return rc_rssi_dbm10(rc_hal_read_status(CC1101_RSSI)); }

static void survey_start(uint16_t secs) {
  rc_hal_gdo_mask();
  cc1101_enter_async_rx();
  sv_capture = 0;
  sv_head = 0;
  sv_triggered = 0;
  sv_dumping = 0;
  sv_bursts = 0;
  TCCR1A = 0;
  TCCR1B = (1 << CS11); /* F_CPU/8: 0.5 us per tick */
  TIFR1 = (1 << TOV1);
  TIMSK1 = (1 << TOIE1);
  EICRA = (uint8_t)((EICRA & (uint8_t) ~((1 << ISC11) | (1 << ISC10))) | (1 << ISC10)); /* INT1 any edge */
  rc.survey_active = 1;
  cli();
  sv_ovf = 2; /* the first run (since now) saturates */
  sv_last_t = TCNT1;
  sv_last_level = (PIND >> PD3) & 1;
  sv_capture = 1;
  sei();
  rc_hal_gdo_unmask(); /* INT1 only while surveying */

  /* Noise floor: median of 40 RSSI readings over ~200 ms. */
  int16_t s[40];
  for (uint8_t i = 0; i < 40; i++) {
    rc_hal_delay_us(5000);
    int16_t v = rssi_dbm10();
    uint8_t j = i;
    for (; j > 0 && s[j - 1] > v; j--) s[j] = s[j - 1];
    s[j] = v;
  }
  sv_thr = (int16_t)(s[20] + RC_SURVEY_MARGIN_DBM10);
  sv_end_us = rc_hal_micros() + (uint32_t)secs * 1000000UL;
  rc_print_survey_on(secs, s[20], sv_thr);
}

static void survey_stop(void) {
  rc_hal_gdo_mask();
  sv_capture = 0;
  rc.survey_active = 0;
  TIMSK1 = 0;
  TCCR1B = 0;
  EICRA = (uint8_t)((EICRA & (uint8_t) ~((1 << ISC11) | (1 << ISC10))) | (1 << ISC11)); /* falling */
  rc_survey_finish(sv_bursts);
}

void rc_hal_survey(uint16_t secs) {
  if (secs)
    survey_start(secs);
  else
    survey_stop();
}

static uint16_t sv_head_now(void) {
  cli();
  uint16_t h = sv_head;
  sei();
  return h;
}

/* One step per main-loop pass: dump a captured burst a line at a time,
 * else watch RSSI for the next one. */
static void survey_tick(void) {
  uint32_t now = rc_hal_micros();
  if (sv_dumping) {
    uint16_t left = (uint16_t)(sv_dump_end - sv_dump_pos);
    if (left) {
      uint8_t n = left > SV_RUNS_PER_LINE ? SV_RUNS_PER_LINE : (uint8_t)left;
      rc_print_raw_start(sv_bursts);
      for (uint8_t i = 0; i < n; i++) {
        uint16_t r = rc.mem.runs[(uint8_t)(sv_dump_pos + i)];
        rc_print_raw_run((uint8_t)(r >> 15), r & 0x7FFF);
      }
      rc_print_eol();
      sv_dump_pos = (uint16_t)(sv_dump_pos + n);
      left = (uint16_t)(left - n);
    }
    if (!left) {
      rc_print_rawe(sv_bursts, sv_dump_runs);
      sv_dumping = 0;
      sv_triggered = 0;
      sv_capture = 1;
    }
    return;
  }
  if (!sv_triggered) {
    if ((int32_t)(now - sv_end_us) >= 0) {
      survey_stop();
      return;
    }
    int16_t dbm = rssi_dbm10();
    if (dbm > sv_thr) {
      sv_triggered = 1;
      sv_trig_us = now;
      sv_trig_head = sv_head_now();
      sv_peak = dbm;
    }
    return;
  }
  int16_t dbm = rssi_dbm10();
  if (dbm > sv_peak) sv_peak = dbm;
  uint16_t head = sv_head_now();
  if ((now - sv_trig_us) < SV_POST_US && (uint16_t)(head - sv_trig_head) < SV_RING - SV_PRE_RUNS) return;
  sv_capture = 0;
  head = sv_head_now();
  uint16_t avail = head < SV_RING ? head : SV_RING;
  uint16_t lo = (uint16_t)(head - avail);
  uint16_t before = (uint16_t)(sv_trig_head - lo); /* > avail: the trigger fell out of the ring */
  if (before <= avail && before > SV_PRE_RUNS) lo = (uint16_t)(sv_trig_head - SV_PRE_RUNS);
  sv_bursts++;
  rc_print_rawb(sv_bursts, sv_peak);
  sv_dump_pos = lo;
  sv_dump_end = head;
  sv_dump_runs = (uint16_t)(head - lo);
  sv_dumping = 1;
}

/* ------------------------------------------------------------- main */

int main(void) {
  uart_init();
  timer0_init();
  spi_init();
  gdo_init();

  /* Interrupts stay off until the CC1101 is configured: GDO2 toggles
   * CHIP_RDYn during SRES, and the GDO handlers do SPI themselves. */
  rc_begin();
  if (!rc_self_test()) rc_print_self_test_failed();
  UCSR0B |= (1 << RXCIE0);
  sei();

  for (;;) {
    if (uart_rx_overrun) {
      cli();
      uart_rx_overrun = 0;
      sei();
      rc_print_overrun();
    }
    char c;
    if (uart_getc(&c))
      rc_feed(c);
    else
      rc_poll();
    rc_service();
    if (rc.survey_active) survey_tick();
  }
}
