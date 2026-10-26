/*
 * cores3_gfx.h - hardware-free drawing for the CoreS3 screen: layout and anti-aliased
 * text composited one row at a time (no frame buffer). Kept free of ESP-IDF headers
 * so the layout can be rendered and checked on a desktop.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "cores3_font.h"

namespace cores3 {

constexpr int SCREEN_W = 320, SCREEN_H = 240;
constexpr int BAND_NAME_Y0 = 0, BAND_TEMP_Y0 = 58, BAND_STATUS_Y0 = 186;
constexpr char DEG = '\x7f';

struct Rgb {
  uint8_t r, g, b;
};
constexpr Rgb BG_IDLE = {14, 30, 58};
constexpr Rgb BG_HEAT = {196, 64, 12};
constexpr Rgb FG_MAIN = {255, 255, 255};
constexpr Rgb FG_DIM_IDLE = {165, 185, 212};
constexpr Rgb FG_DIM_HEAT = {255, 214, 188};

struct Page {
  char name[32] = {0};
  float temp = NAN, setpoint = NAN, power = NAN;
  bool action_heating = false;
  bool heating() const { return action_heating || (power == power && power > 5.0f); }
};

struct Item {
  const Font *font;
  char text[48];
  int x, baseline;
  Rgb color;
};

struct Band {
  int y0, y1;
  Item items[3];
  int n_items = 0;
  bool dots = false;
  bool divider = false; /* dark line along the top edge, between stacked heaters */
  Rgb bg = BG_IDLE;
  char sig[160] = {0};
};

constexpr int MAX_BANDS = 4;

struct Screen {
  Band band[MAX_BANDS];
  int n_bands = 3;
  Rgb bg;
};

inline const Glyph *find_glyph(const Font &f, char c) {
  uint8_t code = (uint8_t) c;
  for (int i = 0; i < f.n; i++)
    if (f.g[i].code == code) return &f.g[i];
  return nullptr;
}

inline int measure(const Font &f, const char *s) {
  int w = 0;
  for (; *s; s++) {
    const Glyph *g = find_glyph(f, *s);
    if (g == nullptr) g = find_glyph(f, '?');
    if (g != nullptr) w += g->adv;
  }
  return w;
}

inline void add_item(Band &b, const Font &f, const char *text, int x, int baseline, Rgb color) {
  Item &it = b.items[b.n_items++];
  it.font = &f;
  snprintf(it.text, sizeof(it.text), "%s", text);
  it.x = x;
  it.baseline = baseline;
  it.color = color;
}

inline void add_centered(Band &b, const Font &f, const char *text, int baseline, Rgb color) {
  add_item(b, f, text, (SCREEN_W - measure(f, text)) / 2, baseline, color);
}

inline void set_sig(Band &b, const Screen &s, int page, int pages) {
  char *p = b.sig;
  size_t left = sizeof(b.sig);
  int w = snprintf(p, left, "%d-%d%d|%u.%u.%u|%d", b.y0, b.y1, b.divider, b.bg.r, b.bg.g, b.bg.b,
                   b.dots ? page * 16 + pages : 0);
  p += w, left -= w;
  for (int i = 0; i < b.n_items && left > 1; i++) {
    w = snprintf(p, left, "|%s@%d", b.items[i].text, b.items[i].x);
    p += w, left -= w;
  }
}

/* Builds the three horizontal bands for one heater page. */
inline void layout(const Page &pg, int page, int pages, Screen &s) {
  const bool heat = pg.heating();
  s.bg = heat ? BG_HEAT : BG_IDLE;
  const Rgb dim = heat ? FG_DIM_HEAT : FG_DIM_IDLE;
  s.n_bands = 3;
  for (Band &b : s.band) b.n_items = 0, b.divider = false, b.bg = s.bg;
  s.band[0].y0 = BAND_NAME_Y0, s.band[0].y1 = BAND_TEMP_Y0;
  s.band[1].y0 = BAND_TEMP_Y0, s.band[1].y1 = BAND_STATUS_Y0;
  s.band[2].y0 = BAND_STATUS_Y0, s.band[2].y1 = SCREEN_H;
  s.band[0].dots = s.band[1].dots = false;
  s.band[2].dots = pages > 1;

  char name[32];
  snprintf(name, sizeof(name), "%s", pg.name[0] ? pg.name : "TermoWeb");
  while (strlen(name) > 1 && measure(FONT_NAME, name) > SCREEN_W - 20) name[strlen(name) - 1] = 0;
  add_centered(s.band[0], FONT_NAME, name, 42, FG_MAIN);

  char num[12];
  if (pg.temp != pg.temp) snprintf(num, sizeof(num), "--.-");
  else snprintf(num, sizeof(num), "%.1f", (double) pg.temp);
  const char unit[3] = {DEG, 'C', 0};
  const int gap = 6;
  const int wn = measure(FONT_BIG, num), wu = measure(FONT_UNIT, unit);
  const int x0 = (SCREEN_W - (wn + gap + wu)) / 2;
  add_item(s.band[1], FONT_BIG, num, x0, 160, FG_MAIN);
  add_item(s.band[1], FONT_UNIT, unit, x0 + wn + gap, 112, dim);

  char status[48], set[20], pow[16];
  if (pg.setpoint != pg.setpoint) snprintf(set, sizeof(set), "Set --");
  else snprintf(set, sizeof(set), "Set %.1f%cC", (double) pg.setpoint, DEG);
  if (heat && pg.power == pg.power && pg.power > 5.0f) snprintf(pow, sizeof(pow), "Heating %.0f W", (double) pg.power);
  else snprintf(pow, sizeof(pow), heat ? "Heating" : "Idle");
  snprintf(status, sizeof(status), "%s  |  %s", pow, set);
  add_centered(s.band[2], FONT_SMALL, status, 212, dim);

  for (int i = 0; i < 3; i++) set_sig(s.band[i], s, page, pages);
}

/* Condensed view: all heaters stacked, each row on its own heating colour. */
inline void layout_overview(const Page *pages, int n, Screen &s) {
  s.n_bands = n;
  const int h = SCREEN_H / n;
  const int margin = 14;
  for (int i = 0; i < n; i++) {
    const Page &pg = pages[i];
    Band &b = s.band[i];
    const bool heat = pg.heating();
    const Rgb dim = heat ? FG_DIM_HEAT : FG_DIM_IDLE;
    b.n_items = 0;
    b.dots = false;
    b.divider = i > 0;
    b.bg = heat ? BG_HEAT : BG_IDLE;
    b.y0 = i * h;
    b.y1 = i == n - 1 ? SCREEN_H : (i + 1) * h;

    char temp[16];
    const char unit[3] = {DEG, 'C', 0};
    if (pg.temp != pg.temp) snprintf(temp, sizeof(temp), "--.-%s", unit);
    else snprintf(temp, sizeof(temp), "%.1f%s", (double) pg.temp, unit);
    const int wt = measure(FONT_NAME, temp);

    const bool two_lines = h >= 70;
    const int name_b = b.y0 + (h >= 100 ? 34 : (two_lines ? 30 : 38));
    const int temp_b = name_b + (h >= 100 ? 10 : 6);
    char name[32];
    snprintf(name, sizeof(name), "%s", pg.name[0] ? pg.name : "TermoWeb");
    while (strlen(name) > 1 && measure(FONT_SMALL, name) > SCREEN_W - 2 * margin - wt - 8) name[strlen(name) - 1] = 0;
    add_item(b, FONT_SMALL, name, margin, name_b, FG_MAIN);
    add_item(b, FONT_NAME, temp, SCREEN_W - margin - wt, temp_b, FG_MAIN);
    if (two_lines) {
      char set[20], pow[16], status[48];
      if (pg.setpoint != pg.setpoint) snprintf(set, sizeof(set), "Set --");
      else snprintf(set, sizeof(set), "Set %.1f%cC", (double) pg.setpoint, DEG);
      if (heat && pg.power == pg.power && pg.power > 5.0f) snprintf(pow, sizeof(pow), "Heating %.0f W", (double) pg.power);
      else snprintf(pow, sizeof(pow), heat ? "Heating" : "Idle");
      snprintf(status, sizeof(status), "%s  |  %s", pow, set);
      add_item(b, FONT_SMALL, status, margin, b.y0 + h - (h >= 100 ? 24 : 16), dim);
    }
    set_sig(b, s, 0, 0);
  }
}

inline void blend(uint8_t *px, Rgb fg, int a) {
  px[0] += ((int) fg.r - px[0]) * a / 255;
  px[1] += ((int) fg.g - px[1]) * a / 255;
  px[2] += ((int) fg.b - px[2]) * a / 255;
}

/* Composes screen row y of a band into rgb[SCREEN_W * 3]. */
inline void compose_row(const Band &b, int y, int page, int pages, uint8_t *rgb) {
  Rgb bg = b.bg;
  if (b.divider && y - b.y0 < 2) bg = {0, 0, 0};
  for (int x = 0; x < SCREEN_W; x++) rgb[x * 3] = bg.r, rgb[x * 3 + 1] = bg.g, rgb[x * 3 + 2] = bg.b;
  for (int i = 0; i < b.n_items; i++) {
    const Item &it = b.items[i];
    int x = it.x;
    for (const char *c = it.text; *c; c++) {
      const Glyph *g = find_glyph(*it.font, *c);
      if (g == nullptr) g = find_glyph(*it.font, '?');
      if (g == nullptr) continue;
      const int gy = y - (it.baseline + g->yo);
      if (gy >= 0 && gy < g->h) {
        const uint8_t *src = it.font->data + g->off + gy * g->w;
        for (int col = 0; col < g->w; col++) {
          const int px = x + g->xo + col;
          if (src[col] && px >= 0 && px < SCREEN_W) blend(rgb + px * 3, it.color, src[col]);
        }
      }
      x += g->adv;
    }
  }
  if (b.dots) {
    const int cy = 230, r = 4, pitch = 18;
    if (y >= cy - r - 1 && y <= cy + r + 1) {
      for (int d = 0; d < pages; d++) {
        const float cx = SCREEN_W / 2.0f + (d - (pages - 1) / 2.0f) * pitch;
        const int a_max = d == page ? 255 : 90;
        for (int x = (int) cx - r - 1; x <= (int) cx + r + 1; x++) {
          const float dist = sqrtf((x + 0.5f - cx) * (x + 0.5f - cx) + (y + 0.5f - cy) * (y + 0.5f - cy));
          float cov = r + 0.5f - dist;
          cov = cov < 0 ? 0 : (cov > 1 ? 1 : cov);
          if (cov > 0 && x >= 0 && x < SCREEN_W) blend(rgb + x * 3, FG_MAIN, (int) (cov * a_max));
        }
      }
    }
  }
}

}  // namespace cores3
