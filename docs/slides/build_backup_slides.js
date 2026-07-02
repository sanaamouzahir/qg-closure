// build_backup_slides.js
//
// Detailed backup deck for the closure NN architecture progression.
// One stage per slide, with explicit inputs and architecture details.
//
//   Stage 0  :  Generic UNet, 2 inputs                   plateau ~45%
//   Stage 1  :  Same UNet, "rich" 6 inputs               plateau ~30%
//   Stage 2  :  Fix C: + analytical N_0, N_dot_0_anal    intermediate
//   Stage 3  :  Fix D: + L^k omega, L^k N (13 ch total)  cleaner target
//   Stage 4  :  Fix D v2 + BilinearClosureNet            plateau 19% (current)
//
// Run:  NODE_PATH=$(npm root -g) node build_backup_slides.js

const pptxgen = require("pptxgenjs");

const C = {
  navy:    "1E2761",
  ice:     "CADCFC",
  white:   "FFFFFF",
  charcoal:"2C2C2C",
  slate:   "5A6B7C",
  amber:   "F4A72A",
  emerald: "1F7A4D",
  red:     "C44747",
  cardBg:  "F7F9FB",
  border:  "D8DEE6",
  monoBg:  "F2F2EE",
};

const FT_TITLE = "Georgia";
const FT_BODY  = "Calibri";
const FT_MONO  = "Consolas";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";  // 13.333 x 7.5

// ---------------------------------------------------------------- helpers --

function header(s, title, subtitle) {
  s.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: 13.333, h: 0.10,
    fill: { color: C.navy }, line: { color: C.navy },
  });
  s.addText(title, {
    x: 0.5, y: 0.20, w: 12.3, h: 0.55,
    fontFace: FT_TITLE, fontSize: 26, bold: true, color: C.navy,
  });
  if (subtitle) {
    s.addText(subtitle, {
      x: 0.5, y: 0.78, w: 12.3, h: 0.32,
      fontFace: FT_BODY, fontSize: 13, italic: true, color: C.slate,
    });
  }
}

function footer(s, pageNum) {
  s.addShape(pres.ShapeType.rect, {
    x: 0, y: 7.4, w: 13.333, h: 0.10,
    fill: { color: C.ice }, line: { color: C.ice },
  });
  s.addText("Closure NN architecture progression  ·  backup slides", {
    x: 0.5, y: 7.15, w: 10, h: 0.25,
    fontFace: FT_BODY, fontSize: 10, italic: true, color: C.slate,
  });
  s.addText(`${pageNum}`, {
    x: 12.5, y: 7.15, w: 0.5, h: 0.25,
    fontFace: FT_BODY, fontSize: 10, color: C.slate, align: "right",
  });
}

function card(s, x, y, w, h, opts = {}) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h,
    fill: { color: opts.bg || C.cardBg },
    line: { color: opts.borderColor || C.border, width: 1 },
    rectRadius: 0.06,
  });
}

function sectionTitle(s, x, y, w, text, color = C.navy) {
  s.addText(text, {
    x, y, w, h: 0.32,
    fontFace: FT_BODY, fontSize: 14, bold: true, color,
  });
}

// =========================================================================
// Slide 1 -- title
// =========================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addShape(pres.ShapeType.rect, {
    x: 0.5, y: 3.05, w: 1.2, h: 0.06,
    fill: { color: C.amber }, line: { color: C.amber },
  });
  s.addText("Closure NN", {
    x: 0.5, y: 2.4, w: 12, h: 0.6,
    fontFace: FT_TITLE, fontSize: 22, italic: true, color: C.ice,
  });
  s.addText("Architecture progression", {
    x: 0.5, y: 3.2, w: 12, h: 1.0,
    fontFace: FT_TITLE, fontSize: 44, bold: true, color: C.white,
  });
  s.addText("From a generic UNet at 45 % rel. L\u00b2 to a numerics-informed " +
            "bilinear closure at 19 % rel. L\u00b2.\n" +
            "Five stages, each addressing a specific failure mode of the previous one.",
    {
      x: 0.5, y: 4.4, w: 11, h: 1.0,
      fontFace: FT_BODY, fontSize: 16, italic: true, color: C.ice,
    });
  s.addText("Backup slides for the QG temporal closure project", {
    x: 0.5, y: 6.8, w: 11, h: 0.3,
    fontFace: FT_BODY, fontSize: 11, color: C.ice,
  });
}

// =========================================================================
// Slide 2 -- the closure formula (context for everything that follows)
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Setup  \u00b7  what the network actually has to learn",
    "Modified-equation analysis gives a closed form. The NN learns ONE term of it.");
  footer(s, 2);

  card(s, 0.5, 1.4, 12.3, 1.7, { bg: "EAF3EF", borderColor: "9DC2B0" });
  s.addText("Full closure increment over one coarse step \u0394T (matched to fine-step ref):", {
    x: 0.7, y: 1.5, w: 12.0, h: 0.32,
    fontFace: FT_BODY, fontSize: 13, bold: true, color: C.emerald,
  });
  s.addText("\u03b4R  =  (\u0394T\u00b3 / 12) \u00b7 ( 1 \u2212 1/K\u00b2 ) \u00b7  [  L\u00b3 \u03c9  +  L\u00b2 N  +  L\u00b7N\u0307  \u2212  5 N\u0308  ]", {
    x: 0.7, y: 1.85, w: 12.0, h: 0.45,
    fontFace: FT_MONO, fontSize: 18, color: C.charcoal,
  });
  s.addText("\u2191 cheap (linear, spectral, free at inference)        \u2191 NN target  \u2014  ANALYTICAL but expensive", {
    x: 0.7, y: 2.40, w: 12.0, h: 0.32,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });
  s.addText("with  N = \u2212J(\u03c8, \u03c9) + F  ,  L = \u03bd\u00b7Lap \u2212 \u03bc \u00b7 I \u2212 \u03b2 \u00b7 \u2202\u2093 \u00b7 Lap\u207b\u00b9", {
    x: 0.7, y: 2.72, w: 12.0, h: 0.32,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });

  card(s, 0.5, 3.3, 6.0, 3.6, {});
  sectionTitle(s, 0.7, 3.4, 5.6, "Why split the closure?");
  s.addText([
    { text: "\u2022 ", options: { bold: true } },
    { text: "L\u00b3\u03c9, L\u00b2N: ", options: { bold: true } },
    { text: "diagonal in Fourier space \u2014 free at inference, also exactly known a priori. No reason to make the NN learn them.\n" },
    { text: "\u2022 ", options: { bold: true } },
    { text: "L\u00b7N\u0307 \u2212 5N\u0308: ", options: { bold: true } },
    { text: "involves time derivatives of the Jacobian J(\u03c8, \u03c9). Computing them analytically would require a few extra Jacobian evaluations PER STEP \u2014 too expensive at inference.\n" },
    { text: "\u2022 ", options: { bold: true } },
    { text: "Decision: ", options: { bold: true } },
    { text: "split  \u03b4R = f_anal + f_NN.  Compute f_anal cheaply at inference, train NN to predict f_NN.\n" },
  ], {
    x: 0.7, y: 3.8, w: 5.6, h: 3.0,
    fontFace: FT_BODY, fontSize: 12, color: C.charcoal, valign: "top",
    paraSpaceAfter: 4,
  });

  card(s, 6.8, 3.3, 6.0, 3.6, { bg: "FFF7E0", borderColor: "E5C97A" });
  sectionTitle(s, 7.0, 3.4, 5.6, "The training target", C.amber);
  s.addText("f_NN_target  =  (1/12) \u00b7 [  L\u00b7N\u0307  \u2212  5 N\u0308  ]", {
    x: 7.0, y: 3.85, w: 5.6, h: 0.5,
    fontFace: FT_MONO, fontSize: 16, bold: true, color: C.charcoal,
  });
  s.addText([
    { text: "\u2022 \u0394T-INDEPENDENT physics quantity (no \u0394T\u00b3 prefactor).\n" },
    { text: "\u2022 Scale O(1) for QG.  No tiny labels, no cancellation issues.\n" },
    { text: "\u2022 Same target reusable across (\u0394T, K) settings: just rescale at inference.\n" },
    { text: "\u2022 N\u0307, N\u0308 are bilinear in \u03c9 and \u03c8 \u2014 hint at the architecture (Stage 4).\n" },
  ], {
    x: 7.0, y: 4.4, w: 5.6, h: 2.4,
    fontFace: FT_BODY, fontSize: 11, color: C.charcoal, valign: "top",
    paraSpaceAfter: 4,
  });
}

// =========================================================================
// Slide 3 -- Stage 0 (baseline)
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Stage 0  \u00b7  Generic UNet, minimal inputs",
    "Sanity-check baseline: the simplest thing that could work");
  footer(s, 3);

  // Inputs card (left)
  card(s, 0.5, 1.4, 4.2, 5.5, {});
  sectionTitle(s, 0.7, 1.5, 4.0, "Inputs (2 channels)");
  s.addText([
    { text: "\u03c9\u2080", options: { fontFace: FT_MONO, fontSize: 16, bold: true, color: C.navy } },
    { text: "  vorticity at t = 0\n", options: { italic: true, color: C.slate, fontSize: 12 } },
    { text: "\u03c8\u2080", options: { fontFace: FT_MONO, fontSize: 16, bold: true, color: C.navy } },
    { text: "  streamfunction at t = 0\n", options: { italic: true, color: C.slate, fontSize: 12 } },
  ], {
    x: 0.7, y: 1.95, w: 4.0, h: 1.5,
    valign: "top",
    paraSpaceAfter: 6,
  });

  card(s, 0.7, 3.4, 3.8, 3.2, { bg: C.monoBg, borderColor: C.border });
  s.addText("Argparse line:\n--input-fields omega_0 psi_0", {
    x: 0.85, y: 3.5, w: 3.6, h: 0.7,
    fontFace: FT_MONO, fontSize: 10, color: C.charcoal, valign: "top",
  });
  s.addText("Target convention (old):", {
    x: 0.85, y: 4.3, w: 3.6, h: 0.32,
    fontFace: FT_BODY, fontSize: 11, bold: true, color: C.slate,
  });
  s.addText("f_NN_target = S\u207b\u00b9(\u0394T)\u00b7e_NN\nMixes physics with numerics.", {
    x: 0.85, y: 4.65, w: 3.6, h: 0.9,
    fontFace: FT_MONO, fontSize: 10, color: C.charcoal, valign: "top",
  });

  // Architecture card (middle)
  card(s, 4.9, 1.4, 4.0, 5.5, {});
  sectionTitle(s, 5.1, 1.5, 3.6, "Architecture");
  s.addText("PeriodicUNet  (~700 K params)", {
    x: 5.1, y: 1.95, w: 3.6, h: 0.32,
    fontFace: FT_BODY, fontSize: 12, bold: true, color: C.charcoal,
  });
  s.addText([
    { text: "in_block:  ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv 2\u219232 + Conv 32\u219232\n", options: { fontFace: FT_MONO } },
    { text: "down1:    ", options: { fontFace: FT_MONO, bold: true } },
    { text: "ConvBlock + AvgPool 2\u00d7 \u2192 64 ch\n", options: { fontFace: FT_MONO } },
    { text: "down2:    ", options: { fontFace: FT_MONO, bold: true } },
    { text: "ConvBlock + AvgPool 2\u00d7 \u2192 128 ch\n", options: { fontFace: FT_MONO } },
    { text: "bottle:   ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv 128\u2192128 + Conv 128\u2192128\n", options: { fontFace: FT_MONO } },
    { text: "up2:      ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Bilinear up + skip + Conv \u2192 64\n", options: { fontFace: FT_MONO } },
    { text: "up1:      ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Bilinear up + skip + Conv \u2192 32\n", options: { fontFace: FT_MONO } },
    { text: "head:     ", options: { fontFace: FT_MONO, bold: true } },
    { text: "1\u00d71 Conv \u2192 1 channel\n", options: { fontFace: FT_MONO } },
    { text: "\nKernel 3\u00d73, circular padding throughout.\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 10, color: C.slate } },
    { text: "Receptive field at full res: ~17 px.\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 10, color: C.slate } },
  ], {
    x: 5.1, y: 2.3, w: 3.7, h: 4.5,
    fontSize: 10, color: C.charcoal, valign: "top",
  });

  // Result card (right)
  card(s, 9.1, 1.4, 3.7, 5.5, { bg: "FBE9E9", borderColor: "E1B7B7" });
  sectionTitle(s, 9.3, 1.5, 3.4, "Result", C.red);
  s.addText("\u224845 %", {
    x: 9.3, y: 2.0, w: 3.4, h: 1.1,
    fontFace: FT_TITLE, fontSize: 56, bold: true, color: C.red,
  });
  s.addText("plateau val rel. L\u00b2 by ep 7", {
    x: 9.3, y: 3.05, w: 3.4, h: 0.4,
    fontFace: FT_BODY, fontSize: 12, italic: true, color: C.slate,
  });
  s.addText([
    { text: "Why it failed\n", options: { bold: true, fontSize: 12 } },
    { text: "\u2022 Only \u03c9\u2080, \u03c8\u2080: no temporal info \u2014 \u03c9\u0307, \u03c9\u0308 unrecoverable.\n" },
    { text: "\u2022 Target mixes physics with numerics: f = S\u207b\u00b9(\u0394T)\u00b7e_NN \u21d2 magnitudes scale with \u0394T\u00b3.\n" },
    { text: "\u2022 UNet downsampling washes out the high-k structure where N\u0308 lives.\n" },
    { text: "\u2022 No structural prior for J(\u03c8, \u03c9) products.\n" },
  ], {
    x: 9.3, y: 3.55, w: 3.4, h: 3.3,
    fontFace: FT_BODY, fontSize: 9, color: C.charcoal, valign: "top",
    paraSpaceAfter: 3,
  });
}

// =========================================================================
// Slide 4 -- Stage 1 (richer inputs, same UNet)
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Stage 1  \u00b7  Same UNet, \u201crich\u201d input set",
    "Add gradients + one previous-step \u03c9 so the network has the building blocks for \u03c9\u0307");
  footer(s, 4);

  card(s, 0.5, 1.4, 4.2, 5.5, {});
  sectionTitle(s, 0.7, 1.5, 4.0, "Inputs (6 channels)");
  s.addText([
    { text: "\u03c9\u2080", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity at t=0\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c8\u2080", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  streamfunction at t=0\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "|\u2207\u03c8|\u00b2", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  kinetic energy density (proxy for u\u00b2+v\u00b2)\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u2202\u2093\u03c9", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity x-gradient\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u2202\u1d67\u03c9", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity y-gradient\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c9\u208b\u2081", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity at t=\u2212\u0394T  ", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: " \u2190 KEY: gives FD access to \u03c9\u0307\n", options: { italic: true, color: C.emerald, fontSize: 10, bold: true } },
  ], {
    x: 0.7, y: 1.95, w: 4.0, h: 3.4,
    valign: "top",
    paraSpaceAfter: 4,
  });
  card(s, 0.7, 5.5, 3.8, 1.2, { bg: C.monoBg, borderColor: C.border });
  s.addText("--input-fields omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1", {
    x: 0.85, y: 5.65, w: 3.6, h: 0.95,
    fontFace: FT_MONO, fontSize: 9, color: C.charcoal, valign: "top",
  });

  card(s, 4.9, 1.4, 4.0, 5.5, {});
  sectionTitle(s, 5.1, 1.5, 3.6, "Architecture");
  s.addText("PeriodicUNet  (same as Stage 0, ~700 K params)", {
    x: 5.1, y: 1.95, w: 3.6, h: 0.5,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });
  s.addText([
    { text: "Only difference from Stage 0:\n", options: { bold: true, fontSize: 11 } },
    { text: "in_block:  ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv ", options: { fontFace: FT_MONO } },
    { text: "6\u219232", options: { fontFace: FT_MONO, bold: true, color: C.emerald } },
    { text: " + Conv 32\u219232\n", options: { fontFace: FT_MONO } },
    { text: "                  (was 2\u219232)\n", options: { fontFace: FT_MONO, italic: true, color: C.slate } },
    { text: "\nEverything else identical: 2 down + bottleneck + 2 up + 1\u00d71 head.\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 10, color: C.slate } },
    { text: "\nTarget still S\u207b\u00b9(\u0394T)\u00b7e_NN (NOT yet decoupled).\n",
      options: { fontFace: FT_BODY, fontSize: 10, color: C.red, italic: true } },
  ], {
    x: 5.1, y: 2.5, w: 3.7, h: 4.3,
    fontSize: 11, color: C.charcoal, valign: "top",
  });

  card(s, 9.1, 1.4, 3.7, 5.5, { bg: "FFF7E0", borderColor: "E5C97A" });
  sectionTitle(s, 9.3, 1.5, 3.4, "Result", C.amber);
  s.addText("\u224830 %", {
    x: 9.3, y: 2.0, w: 3.4, h: 1.1,
    fontFace: FT_TITLE, fontSize: 56, bold: true, color: C.amber,
  });
  s.addText("plateau val rel. L\u00b2  (\u221215 pp vs Stage 0)", {
    x: 9.3, y: 3.05, w: 3.4, h: 0.4,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });
  s.addText([
    { text: "Lessons\n", options: { bold: true, fontSize: 12 } },
    { text: "\u2022 More inputs help: temporal info via \u03c9\u208b\u2081 alone gives \u22125 pp.\n" },
    { text: "\u2022 Gradients (\u2202\u2093\u03c9, \u2202\u1d67\u03c9, |\u2207\u03c8|\u00b2) help the network discover Jacobian-like products.\n" },
    { text: "\u2022 Still bottlenecked: target is still \u0394T-coupled, no \u03c9\u208b\u2082 for \u03c9\u0308 stencil.\n" },
  ], {
    x: 9.3, y: 3.55, w: 3.4, h: 3.3,
    fontFace: FT_BODY, fontSize: 9, color: C.charcoal, valign: "top",
    paraSpaceAfter: 3,
  });
}

// =========================================================================
// Slide 5 -- Stage 2 (Fix C, more analytical inputs)
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Stage 2  \u00b7  Fix C  \u2014  expose analytical building blocks",
    "Hand the network N and \u1e44 directly so it doesn\u2019t have to rediscover them");
  footer(s, 5);

  card(s, 0.5, 1.4, 4.2, 5.5, {});
  sectionTitle(s, 0.7, 1.5, 4.0, "Inputs (8 channels)");
  s.addText([
    { text: "\u2014 the 6 from Stage 1: ", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c9\u2080, \u03c8\u2080, |\u2207\u03c8|\u00b2, \u2202\u2093\u03c9, \u2202\u1d67\u03c9, \u03c9\u208b\u2081\n",
      options: { fontFace: FT_MONO, fontSize: 11 } },
    { text: "\nNEW analytical channels:\n",
      options: { bold: true, fontSize: 11, color: C.emerald } },
    { text: "N\u2080", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  the nonlinear RHS at t=0:  N = \u2212J(\u03c8, \u03c9) + F\n",
      options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u1e44\u2080", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  ANALYTICAL time deriv of N at t=0,\n",
      options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "      via chain rule + bilinearity of J\n",
      options: { italic: true, color: C.slate, fontSize: 10 } },
  ], {
    x: 0.7, y: 1.95, w: 4.0, h: 3.5,
    valign: "top",
    paraSpaceAfter: 4,
  });
  card(s, 0.7, 5.55, 3.8, 1.2, { bg: C.monoBg, borderColor: C.border });
  s.addText("--input-fields omega_0 psi_0 grad_psi_sq omega_x omega_y omega_m1\n                N_0 N_dot_0_anal", {
    x: 0.85, y: 5.65, w: 3.6, h: 1.0,
    fontFace: FT_MONO, fontSize: 9, color: C.charcoal, valign: "top",
  });

  card(s, 4.9, 1.4, 4.0, 5.5, {});
  sectionTitle(s, 5.1, 1.5, 3.6, "Architecture");
  s.addText("PeriodicUNet (~700 K params)  \u2014  unchanged except in_block:", {
    x: 5.1, y: 1.95, w: 3.6, h: 0.55,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });
  s.addText([
    { text: "in_block:  ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv ", options: { fontFace: FT_MONO } },
    { text: "8\u219232", options: { fontFace: FT_MONO, bold: true, color: C.emerald } },
    { text: " + Conv 32\u219232\n", options: { fontFace: FT_MONO } },
  ], {
    x: 5.1, y: 2.55, w: 3.7, h: 0.5,
    fontSize: 11, color: C.charcoal, valign: "top",
  });
  card(s, 5.1, 3.2, 3.6, 3.5, { bg: "FFF7E0", borderColor: "E5C97A" });
  sectionTitle(s, 5.25, 3.3, 3.4, "Trade-off", C.amber);
  s.addText([
    { text: "\u1e44 inputs are ANALYTICAL \u2014 computed from \u03c9\u2080 alone via chain rule.\n" },
    { text: "\u2022 Pro: the network sees the EXACT building blocks of the closure target N\u0308 = ...\n" },
    { text: "\u2022 Con: those analytical inputs cost extra Jacobian evaluations. If we ship this at inference, we eat that compute.\n" },
    { text: "\u2022 Used here as DIAGNOSTIC: tells us if the gap was in feature construction or in regression.\n" },
  ], {
    x: 5.25, y: 3.65, w: 3.3, h: 2.95,
    fontFace: FT_BODY, fontSize: 10, color: C.charcoal, valign: "top",
    paraSpaceAfter: 3,
  });

  card(s, 9.1, 1.4, 3.7, 5.5, { bg: "FFF7E0", borderColor: "E5C97A" });
  sectionTitle(s, 9.3, 1.5, 3.4, "Result", C.amber);
  s.addText("\u2248 28 %", {
    x: 9.3, y: 2.0, w: 3.4, h: 1.1,
    fontFace: FT_TITLE, fontSize: 50, bold: true, color: C.amber,
  });
  s.addText("plateau val rel. L\u00b2  (\u22122 pp vs Stage 1)", {
    x: 9.3, y: 3.05, w: 3.4, h: 0.4,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });
  s.addText([
    { text: "Diagnosis\n", options: { bold: true, fontSize: 12 } },
    { text: "\u2022 The remaining gap is NOT just feature construction. Even with N\u0307 handed to it, the network plateaus.\n" },
    { text: "\u2022 \u2192 the bottleneck is in REGRESSION  (architecture or target).  Move to Fix D.\n" },
  ], {
    x: 9.3, y: 3.55, w: 3.4, h: 3.3,
    fontFace: FT_BODY, fontSize: 9, color: C.charcoal, valign: "top",
    paraSpaceAfter: 3,
  });
}

// =========================================================================
// Slide 6 -- Stage 3 (Fix D, full analytical scaffolding + clean target)
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Stage 3  \u00b7  Fix D  \u2014  decouple physics from numerics",
    "Clean the target. Then expose every analytical scaffolding term as input.");
  footer(s, 6);

  // The KEY change: target switch
  card(s, 0.5, 1.3, 12.3, 1.5, { bg: "EAF3EF", borderColor: "9DC2B0" });
  s.addText("Old target", {
    x: 0.7, y: 1.42, w: 2.0, h: 0.32,
    fontFace: FT_BODY, fontSize: 11, bold: true, color: C.slate,
  });
  s.addText("f_NN = S\u207b\u00b9(\u0394T)\u00b7e_NN", {
    x: 0.7, y: 1.72, w: 5.5, h: 0.45,
    fontFace: FT_MONO, fontSize: 14, color: C.charcoal,
  });
  s.addText("\u2192", {
    x: 6.2, y: 1.85, w: 0.4, h: 0.45,
    fontFace: FT_TITLE, fontSize: 26, bold: true, color: C.emerald,
  });
  s.addText("New target  (\u0394T-independent)", {
    x: 6.7, y: 1.42, w: 5.5, h: 0.32,
    fontFace: FT_BODY, fontSize: 11, bold: true, color: C.emerald,
  });
  s.addText("f_NN = (1/12)\u00b7[ L\u00b7N\u0307 \u2212 5N\u0308 ]", {
    x: 6.7, y: 1.72, w: 5.8, h: 0.45,
    fontFace: FT_MONO, fontSize: 14, color: C.charcoal,
  });
  s.addText("\u0394T\u00b3, K, S\u207b\u00b9 recovered analytically at inference.  Target is now O(1) physics.", {
    x: 0.7, y: 2.32, w: 12.0, h: 0.32,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });

  card(s, 0.5, 2.95, 4.4, 4.0, {});
  sectionTitle(s, 0.7, 3.05, 4.2, "Inputs (13 channels)");
  s.addText([
    { text: "Stage 2's 8: ", options: { italic: true, color: C.slate, fontSize: 10 } },
    { text: "\u03c9\u2080, \u03c8\u2080, |\u2207\u03c8|\u00b2, \u2202\u2093\u03c9, \u2202\u1d67\u03c9, \u03c9\u208b\u2081, N\u2080, \u1e44\u2080\n",
      options: { fontFace: FT_MONO, fontSize: 9 } },
    { text: "\nPLUS analytical scaffolding:\n",
      options: { bold: true, fontSize: 11, color: C.emerald } },
    { text: "L\u03c9\u2080 ", options: { fontFace: FT_MONO, fontSize: 12, bold: true, color: C.navy } },
    { text: " linear operator on \u03c9\n",
      options: { italic: true, color: C.slate, fontSize: 10 } },
    { text: "L\u00b2\u03c9\u2080 ", options: { fontFace: FT_MONO, fontSize: 12, bold: true, color: C.navy } },
    { text: " (computed via L\u00b7L\u03c9 \u2014 spectral)\n",
      options: { italic: true, color: C.slate, fontSize: 10 } },
    { text: "L\u00b3\u03c9\u2080 ", options: { fontFace: FT_MONO, fontSize: 12, bold: true, color: C.navy } },
    { text: " factor of f_anal\n",
      options: { italic: true, color: C.slate, fontSize: 10 } },
    { text: "LN\u2080  ", options: { fontFace: FT_MONO, fontSize: 12, bold: true, color: C.navy } },
    { text: " linear op on the nonlinear RHS\n",
      options: { italic: true, color: C.slate, fontSize: 10 } },
    { text: "L\u00b2N\u2080 ", options: { fontFace: FT_MONO, fontSize: 12, bold: true, color: C.navy } },
    { text: " other factor of f_anal\n",
      options: { italic: true, color: C.slate, fontSize: 10 } },
  ], {
    x: 0.7, y: 3.5, w: 4.0, h: 3.3,
    valign: "top",
    paraSpaceAfter: 3,
  });

  card(s, 5.0, 2.95, 4.0, 4.0, {});
  sectionTitle(s, 5.2, 3.05, 3.6, "Architecture");
  s.addText([
    { text: "PeriodicUNet (~700 K params, unchanged except in_block):\n\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 10, color: C.slate } },
    { text: "in_block: ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv ", options: { fontFace: FT_MONO } },
    { text: "13\u219232", options: { fontFace: FT_MONO, bold: true, color: C.emerald } },
    { text: " + Conv 32\u219232\n", options: { fontFace: FT_MONO } },
    { text: "\nDownsampling, bottleneck, upsampling: identical to Stage 0.",
      options: { italic: true, fontSize: 10, color: C.slate } },
  ], {
    x: 5.2, y: 3.5, w: 3.6, h: 3.3,
    fontSize: 11, color: C.charcoal, valign: "top",
  });

  card(s, 9.1, 2.95, 3.7, 4.0, { bg: "FFF7E0", borderColor: "E5C97A" });
  sectionTitle(s, 9.3, 3.05, 3.4, "Result", C.amber);
  s.addText("\u224825 %", {
    x: 9.3, y: 3.45, w: 3.4, h: 0.95,
    fontFace: FT_TITLE, fontSize: 46, bold: true, color: C.amber,
  });
  s.addText("plateau val rel. L\u00b2", {
    x: 9.3, y: 4.4, w: 3.4, h: 0.32,
    fontFace: FT_BODY, fontSize: 10, italic: true, color: C.slate,
  });
  s.addText([
    { text: "\u2022 Cleaner target \u2192 stabler training, lower plateau.\n" },
    { text: "\u2022 But UNet still wrong inductive bias for J(\u03c8, \u03c9): hierarchical scales aren't the right structure.\n" },
    { text: "\u2022 Next: change the architecture itself.\n" },
  ], {
    x: 9.3, y: 4.75, w: 3.4, h: 2.1,
    fontFace: FT_BODY, fontSize: 9, color: C.charcoal, valign: "top",
    paraSpaceAfter: 3,
  });
}

// =========================================================================
// Slide 7 -- Stage 4 (Fix D v2, BilinearClosureNet, current)
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Stage 4  \u00b7  Numerics-informed BilinearClosureNet",
    "Custom architecture matching the closure operator structure. CURRENT BEST.");
  footer(s, 7);

  // Inputs card: spatially small (3 cols)
  card(s, 0.5, 1.3, 4.2, 5.7, {});
  sectionTitle(s, 0.7, 1.4, 4.0, "Inputs (6 channels)");
  s.addText("Drop the analytical scaffolding. Replace it with two more time levels:", {
    x: 0.7, y: 1.78, w: 4.0, h: 0.6,
    fontFace: FT_BODY, fontSize: 11, italic: true, color: C.slate,
  });
  s.addText([
    { text: "\u03c9\u2080", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity at t = 0\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c9\u208b\u2081", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity at t = \u2212\u0394T\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c9\u208b\u2082", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  vorticity at t = \u22122\u0394T  ",
      options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u2190 NEW: enables FD \u03c9\u0308\n",
      options: { italic: true, color: C.emerald, fontSize: 10, bold: true } },
    { text: "\u03c8\u2080", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  streamfunction at t = 0\n", options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c8\u208b\u2081", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  streamfunction at t = \u2212\u0394T  \n",
      options: { italic: true, color: C.slate, fontSize: 11 } },
    { text: "\u03c8\u208b\u2082", options: { fontFace: FT_MONO, fontSize: 14, bold: true, color: C.navy } },
    { text: "  streamfunction at t = \u22122\u0394T\n", options: { italic: true, color: C.slate, fontSize: 11 } },
  ], {
    x: 0.7, y: 2.4, w: 4.0, h: 3.0,
    valign: "top",
    paraSpaceAfter: 4,
  });
  card(s, 0.7, 5.55, 3.8, 1.4, { bg: C.monoBg, borderColor: C.border });
  s.addText("--input-fields\n  omega_0 omega_m1 omega_m2\n  psi_0   psi_m1   psi_m2", {
    x: 0.85, y: 5.65, w: 3.6, h: 1.2,
    fontFace: FT_MONO, fontSize: 10, color: C.charcoal, valign: "top",
  });

  // Architecture card (middle)
  card(s, 4.9, 1.3, 4.7, 5.7, {});
  sectionTitle(s, 5.1, 1.4, 4.5, "Architecture (\u223c370 K params)");
  s.addText("BilinearClosureNet \u2014 5 conv blocks, 7-px receptive field", {
    x: 5.1, y: 1.78, w: 4.5, h: 0.32,
    fontFace: FT_BODY, fontSize: 10, italic: true, color: C.slate,
  });
  s.addText([
    { text: "Stem    ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv2d(6 \u2192 64, 1\u00d71)\n", options: { fontFace: FT_MONO } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "channel mix from raw inputs\n\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.slate } },

    { text: "Block A ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv2d(64 \u2192 64, 1\u00d71)\n", options: { fontFace: FT_MONO } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "discovers FD time derivatives \u2014 no spatial coupling\n\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.slate } },

    { text: "Block B ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv2d(64 \u2192 256, 3\u00d73) + GLU \u2192 128\n", options: { fontFace: FT_MONO } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "spatial gradients (3\u00d73) + bilinear products via GLU\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.slate } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "this is the J(\u03c8, \u03c9) approximator\n\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.emerald, bold: true } },

    { text: "Block C ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv2d(128 \u2192 128, 3\u00d73) + GeLU + residual\n", options: { fontFace: FT_MONO } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "refinement (skip connection helps gradients)\n\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.slate } },

    { text: "Block D ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv2d(128 \u2192 64, 3\u00d73) + GeLU\n", options: { fontFace: FT_MONO } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "applies the linear operator L (Laplacian)\n\n",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.slate } },

    { text: "Output  ", options: { fontFace: FT_MONO, bold: true } },
    { text: "Conv2d(64 \u2192 1, 1\u00d71)\n", options: { fontFace: FT_MONO } },
    { text: "        ", options: { fontFace: FT_MONO } },
    { text: "predicts (1/12)[L\u00b7N\u0307 \u2212 5N\u0308]",
      options: { fontFace: FT_BODY, italic: true, fontSize: 9, color: C.slate } },
  ], {
    x: 5.1, y: 2.15, w: 4.5, h: 4.8,
    fontSize: 9, color: C.charcoal, valign: "top",
  });

  // Result card (right)
  card(s, 9.8, 1.3, 3.0, 2.4, { bg: "DCEFE6", borderColor: C.emerald });
  sectionTitle(s, 9.95, 1.42, 2.8, "Result", C.emerald);
  s.addText("19 %", {
    x: 9.95, y: 1.85, w: 2.7, h: 1.0,
    fontFace: FT_TITLE, fontSize: 50, bold: true, color: C.emerald,
  });
  s.addText("plateau val rel. L\u00b2 (current best)", {
    x: 9.95, y: 2.85, w: 2.7, h: 0.7,
    fontFace: FT_BODY, fontSize: 9, italic: true, color: C.slate,
  });

  card(s, 9.8, 3.85, 3.0, 3.15, {});
  sectionTitle(s, 9.95, 3.97, 2.7, "Why this works");
  s.addText([
    { text: "\u2022 \u03c8 is given as input \u2192 inverse-Laplacian \u2207\u207b\u00b2 doesn't have to be learned. No global op needed.\n" },
    { text: "\u2022 Receptive field 7 px matches the closure structure: J is pointwise, gradients are 3\u00d73, L is 3\u00d73.\n" },
    { text: "\u2022 GLU = bilinear products \u2192 J(\u03c8, \u03c9) discovered structurally, not memorized.\n" },
    { text: "\u2022 Circular padding everywhere (domain doubly periodic).\n" },
    { text: "\u2022 No UNet hierarchy: closure has no multi-scale interaction.\n" },
  ], {
    x: 9.95, y: 4.32, w: 2.75, h: 2.65,
    fontFace: FT_BODY, fontSize: 8, color: C.charcoal, valign: "top",
    paraSpaceAfter: 2,
  });
}

// =========================================================================
// Slide 8 -- Summary table + commentary
// =========================================================================
{
  const s = pres.addSlide();
  header(s,
    "Summary  \u00b7  five stages, 26 pp rel. L\u00b2 reduction",
    "Each change addresses a specific failure mode of the previous stage");
  footer(s, 8);

  const head = (t) => ({ text: t,
    options: { bold: true, color: C.white,
               fill: { color: C.navy }, align: "center", valign: "middle",
               fontFace: FT_BODY, fontSize: 11 } });
  const cell = (t, opts={}) => ({ text: t,
    options: { align: "center", valign: "middle",
               fontFace: FT_BODY, fontSize: 10,
               color: C.charcoal, ...opts } });

  const tableData = [
    [ head("Stage"), head("Architecture"), head("Inputs (count)"),
      head("Target"), head("Plateau"), head("Key fix") ],
    [ cell("0"),
      cell("PeriodicUNet"),
      cell("\u03c9\u2080, \u03c8\u2080  (2)"),
      cell("S\u207b\u00b9\u00b7e_NN"),
      cell("\u224845 %", { color: C.red, bold: true }),
      cell("baseline") ],
    [ cell("1"),
      cell("PeriodicUNet"),
      cell("+ |\u2207\u03c8|\u00b2, \u2202\u2093\u03c9, \u2202\u1d67\u03c9, \u03c9\u208b\u2081  (6)"),
      cell("S\u207b\u00b9\u00b7e_NN"),
      cell("\u224830 %", { color: C.amber, bold: true }),
      cell("temporal info") ],
    [ cell("2"),
      cell("PeriodicUNet"),
      cell("+ N\u2080, \u1e44\u2080  (8)"),
      cell("S\u207b\u00b9\u00b7e_NN"),
      cell("\u224828 %", { color: C.amber, bold: true }),
      cell("analytical scaffolding") ],
    [ cell("3"),
      cell("PeriodicUNet"),
      cell("+ L\u00b9\u208b\u00b3\u03c9\u2080, LN\u2080, L\u00b2N\u2080  (13)"),
      cell("(1/12)[L\u00b7\u1e44 \u2212 5N\u0308]",
        { fontFace: FT_MONO, fontSize: 9 }),
      cell("\u224825 %", { color: C.amber, bold: true }),
      cell("clean target") ],
    [ cell("4", { fill: { color: "DCEFE6" }, bold: true }),
      cell("BilinearClosureNet",
        { fill: { color: "DCEFE6" }, bold: true, color: C.emerald }),
      cell("\u03c9\u208b\u2080\u208b\u2081\u208b\u2082, \u03c8\u208b\u2080\u208b\u2081\u208b\u2082  (6)",
        { fill: { color: "DCEFE6" } }),
      cell("(1/12)[L\u00b7\u1e44 \u2212 5N\u0308]",
        { fontFace: FT_MONO, fontSize: 9, fill: { color: "DCEFE6" } }),
      cell("19 %",
        { color: C.emerald, bold: true, fill: { color: "DCEFE6" }, fontSize: 14 }),
      cell("matched architecture",
        { fill: { color: "DCEFE6" }, bold: true, color: C.emerald }) ],
  ];

  s.addTable(tableData, {
    x: 0.4, y: 1.45, w: 12.5,
    colW: [0.7, 2.4, 3.4, 2.4, 1.4, 2.2],
    rowH: 0.5,
    border: { type: "solid", color: C.border, pt: 0.75 },
    fontFace: FT_BODY,
  });

  card(s, 0.5, 5.4, 12.3, 1.6, { bg: C.cardBg });
  sectionTitle(s, 0.7, 5.5, 12.0, "Reading the progression");
  s.addText([
    { text: "\u2022 0 \u2192 1 (\u221215 pp): more inputs help \u2014 the network gains FD access to time derivatives.\n" },
    { text: "\u2022 1 \u2192 2 (\u22122 pp): pre-computing N, \u1e44 helps a little; remaining gap is in regression, not features.\n" },
    { text: "\u2022 2 \u2192 3 (\u22123 pp): cleaner training target (\u0394T-independent physics) \u2014 magnitudes O(1), not O(\u0394T\u00b3).\n" },
    { text: "\u2022 3 \u2192 4 (\u22126 pp): purpose-built architecture \u2014 GLU bilinear products + 7-px local kernels match the closure structure.",
      options: { bold: true, color: C.emerald } },
  ], {
    x: 0.7, y: 5.85, w: 12.0, h: 1.3,
    fontFace: FT_BODY, fontSize: 11, color: C.charcoal, valign: "top",
    paraSpaceAfter: 2,
  });
}

pres.writeFile({ fileName: "closure_nn_progression.pptx" })
    .then(fn => console.log(`wrote ${fn}`));
