// TwinRead status display for the Arduino UNO Q (STM32U585 MCU, 8x13 LED matrix).
// Research prototype - not a diagnostic device.
// The Linux side calls the Bridge RPC method "set_state" with one string:
//   idle | busy | agree | review | no_tissue | error
#include <Arduino_RouterBridge.h>
#include <Arduino_LED_Matrix.h>

Arduino_LED_Matrix matrix;

static const int W = 13, H = 8, NPIX = W * H;  // 104 pixels, row-major

// Icons are 8 rows of 13 characters, '#' = LED on.
static const char *const ICON_IDLE[H] = {
  ".............", ".............", ".............", "......##.....",
  "......##.....", ".............", ".............", "............."};
static const char *const ICON_CHECK[H] = {
  ".............", "...........#.", "..........##.", "#........##..",
  ".##.....##...", "..##...##....", "...##.##.....", "....###......"};
static const char *const ICON_QMARK[H] = {
  "....#####....", "...##...##...", "........##...", ".......##....",
  "......##.....", "......##.....", ".............", "......##....."};
static const char *const ICON_DASH[H] = {
  ".............", ".............", ".............", ".############",
  ".############", ".............", ".............", "............."};
static const char *const ICON_ERROR[H] = {
  "#...........#", ".#.........#.", "..#.......#..", "...#.....#...",
  "....#...#....", ".....#.#.....", "......#......", "............."};
// Spinner: a 3x3 dot orbiting the centre, 8 positions.
static const int SPIN_XY[8][2] = {{5,1},{7,1},{8,3},{7,5},{5,5},{4,3},{4,2},{5,2}};

enum State { S_IDLE, S_BUSY, S_AGREE, S_REVIEW, S_NO_TISSUE, S_ERROR };
volatile int g_state = S_IDLE;  // written by the Bridge callback, read by loop()

void drawIcon(const char *const rows[H]) {
  uint8_t px[NPIX];
  for (int y = 0; y < H; y++)
    for (int x = 0; x < W; x++) px[y * W + x] = (rows[y][x] == '#') ? 1 : 0;
  matrix.loadPixels(px, NPIX);
}

void drawSpinner(int step) {
  uint8_t px[NPIX] = {0};
  for (int k = 0; k < 3; k++) {  // a short comet tail
    int i = (step + 8 - k) % 8;
    px[SPIN_XY[i][1] * W + SPIN_XY[i][0] + 0] = 1;
  }
  matrix.loadPixels(px, NPIX);
}

void set_state(String s) {
  if (s == "idle") g_state = S_IDLE;
  else if (s == "busy") g_state = S_BUSY;
  else if (s == "agree") g_state = S_AGREE;
  else if (s == "review") g_state = S_REVIEW;
  else if (s == "no_tissue") g_state = S_NO_TISSUE;
  else g_state = S_ERROR;
}

void setup() {
  matrix.begin();
  Bridge.begin();
  Bridge.provide("set_state", set_state);
  drawIcon(ICON_IDLE);
}

void loop() {
  static int last = -1, step = 0;
  int s = g_state;
  if (s == S_BUSY) { drawSpinner(step++ & 7); delay(90); return; }
  if (s != last) {
    switch (s) {
      case S_AGREE: drawIcon(ICON_CHECK); break;
      case S_REVIEW: drawIcon(ICON_QMARK); break;
      case S_NO_TISSUE: drawIcon(ICON_DASH); break;
      case S_ERROR: drawIcon(ICON_ERROR); break;
      default: drawIcon(ICON_IDLE);
    }
    last = s;
  }
  delay(20);
}
