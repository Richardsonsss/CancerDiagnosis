"""Diagrams for the README.

    python docs/source/make_diagrams.py        ->  docs/images/en/*.png

Drawn 800 px wide (x3 for print sharpness); scaled to page width, text comes out at about 9 pt.
The app screenshots (docs/images/app_start.png, app_result.png) are taken from the real web app.
"""
import math
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES = os.path.join(os.path.dirname(HERE), "images")
S = 3
W = 800
FONTS = {"regular": r"C:\Windows\Fonts\msyh.ttc", "bold": r"C:\Windows\Fonts\msyhbd.ttc"}

TEAL, TEAL_BG = (15, 118, 110), (230, 244, 242)
BLUE, BLUE_BG = (37, 99, 235), (232, 240, 254)
AMBER, AMBER_BG = (180, 83, 9), (254, 243, 224)
VIOLET, VIOLET_BG = (109, 40, 217), (241, 236, 254)
GREY, INK, MUTED = (100, 116, 139), (15, 23, 42), (71, 85, 105)


def font(size, bold=False):
    return ImageFont.truetype(FONTS["bold" if bold else "regular"], int(size * S))


def canvas(h):
    im = Image.new("RGB", (W * S, int(h * S)), "white")
    return im, ImageDraw.Draw(im)


def text(d, xy, s, size=13, bold=False, fill=INK, anchor="lm"):
    d.text((xy[0] * S, xy[1] * S), s, font=font(size, bold), fill=fill, anchor=anchor)


def arrow(d, x1, y1, x2, y2, color=GREY, both=False, width=3, label="", label_side=1):
    d.line([x1 * S, y1 * S, x2 * S, y2 * S], fill=color, width=width * S)

    def head(xa, ya, xb, yb):
        ang = math.atan2(yb - ya, xb - xa)
        d.polygon([(xb * S, yb * S)] + [((xb - 12 * math.cos(ang + s * 0.5)) * S, (yb - 12 * math.sin(ang + s * 0.5)) * S)
                                        for s in (-1, 1)], fill=color)
    head(x1, y1, x2, y2)
    if both:
        head(x2, y2, x1, y1)
    if label:
        if abs(x2 - x1) > abs(y2 - y1):  # horizontal: label above
            text(d, ((x1 + x2) / 2, min(y1, y2) - 12), label, 11, fill=MUTED, anchor="mm")
        else:  # vertical: label to the right
            text(d, (max(x1, x2) + 12 * label_side, (y1 + y2) / 2), label, 12, fill=MUTED,
                 anchor="lm" if label_side > 0 else "rm")


def panel(d, x, y, w, h, title, fg, bg, title_size=15):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=12 * S, fill=bg, outline=fg, width=2 * S)
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + 34) * S], radius=12 * S, fill=fg)
    d.rectangle([x * S, (y + 22) * S, (x + w) * S, (y + 34) * S], fill=fg)
    text(d, (x + w / 2, y + 17), title, title_size, True, "white", "mm")


def items(d, x, y, rows, fg, width=None, step=44):
    """(item, detail) rows with a coloured dot; detail may be empty."""
    for item, detail in rows:
        d.ellipse([x * S, (y - 3.5) * S, (x + 7) * S, (y + 3.5) * S], fill=fg)
        text(d, (x + 14, y), item, 13.5, True)
        if detail:
            text(d, (x + 14, y + 19), detail, 11.5, fill=MUTED)
        y += step if detail else 26
    return y


def chip(d, x, y, w, h, s, fg, bg, size=13, bold=False):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + h) * S], radius=9 * S, fill=bg, outline=fg, width=2 * S)
    text(d, (x + w / 2, y + h / 2), s, size, bold, INK, "mm")


def code(d, x, y, w, s, size=12):
    d.rounded_rectangle([x * S, y * S, (x + w) * S, (y + 26) * S], radius=6 * S, fill=(241, 245, 249),
                        outline=(203, 213, 225), width=1 * S)
    d.text(((x + 10) * S, (y + 13) * S), s, font=ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", int(size * S)),
           fill=(15, 81, 50), anchor="lm")


# ---------------------------------------------------------------- texts
T = {
    "en": {
        "train": "Training back end · any Linux VM with a GPU",
        "train_items": [[("1  Understand the prompt", "any language, local language model"),
                         ("2  Download the public dataset", "train / validation / test split"),
                         ("3  Screen candidate models", "37 core + newest on Hugging Face")],
                        [("4  Train the best 5", "that can be exported to ONNX"),
                         ("5  Choose voting on validation", "majority / Borda / exponential"),
                         ("6  Export ONNX + INT8", "model package <type>_package.zip")]],
        "pkg": "model package (copy or S3)",
        "serve": "Diagnosis service · Docker server or local computer",
        "serve_items": [[("FastAPI + ONNX Runtime", "GPU (CUDA / DirectML) or CPU + INT8"),
                         ("Voting + result text", "risk level and text only, no numbers")],
                        [("Offline local-network mode", "no internet, QR code for the phone"),
                         ("React PWA", "served by the same service")]],
        "link": "HTTPS or local network",
        "browser": "Browser · phone or computer",
        "browser_items": [[("Choose type, take a photo", "the phone camera opens directly")],
                          [("Text result below the photo", "result and advice")]],
        "flow_title": "Training pipeline",
        "flow": ["Prompt", "Understand", "Pick dataset", "Download", "Screen models", "Train top 5", "Choose voting",
                 "Export package"],
        "cmd_title": "Commands on the training VM",
        "server_boxes": ["Training VM", "Docker server", "Reverse proxy", "Phones & computers"],
        "server_sub": ["artifacts/*_package.zip", "models/  +  container", "HTTPS :443 → :8000", "browser"],
        "server_links": ["copy / S3", "", "HTTPS"],
        "lan_pc": "Local computer",
        "lan_pc_items": [("models/*_package.zip", ""), ("lan/start_lan.ps1", ""), ("no internet needed", "")],
        "lan_phone": "iPhone",
        "lan_phone_items": [("scan the QR code", ""), ("take a photo", ""), ("result in 1-2 s", "")],
        "lan_net": "Same local network - any one of:",
        "lan_nets": ["Wi-Fi router (no internet)", "Computer's mobile hotspot", "iPhone Personal Hotspot"],
        "lan_steps": ["1  Once, while online", "2  Once, as administrator", "3  Every time"],
        "where": [("Training VM", ["training/", "common/"], "train models"),
                  ("Docker server", ["Dockerfile, docker-compose.yml", "server/  web/  common/", "models/"], "public service"),
                  ("Local computer", ["server/  common/", "web/dist  lan/", "models/"], "offline local network")],
        "where_title": "What goes on which computer",
        "screens": ["1  Choose the type, take a photo", "2  The result appears below the photo"],
    },
}


def overview(t, out):
    im, d = canvas(528)
    panel(d, 20, 10, W - 40, 176, t["train"], TEAL, TEAL_BG)
    for i, col in enumerate(t["train_items"]):
        items(d, 40 + i * 380, 64, col, TEAL)
    arrow(d, W / 2, 190, W / 2, 232, label=t["pkg"])
    panel(d, 20, 236, W - 40, 132, t["serve"], BLUE, BLUE_BG)
    for i, col in enumerate(t["serve_items"]):
        items(d, 40 + i * 380, 290, col, BLUE)
    arrow(d, W / 2, 372, W / 2, 414, both=True, label=t["link"])
    panel(d, 20, 418, W - 40, 96, t["browser"], AMBER, AMBER_BG)
    for i, col in enumerate(t["browser_items"]):
        items(d, 40 + i * 380, 472, col, AMBER)
    im.save(out)


def training(t, out):
    im, d = canvas(330)
    text(d, (W / 2, 16), t["flow_title"], 16, True, TEAL, "mm")
    bw, gap = (W - 40 - 3 * 26) / 4, 26
    for i, s in enumerate(t["flow"]):
        r, c = divmod(i, 4)
        c = c if r == 0 else 3 - c
        x, y = 20 + c * (bw + gap), 38 + r * 84
        chip(d, x, y, bw, 48, s, TEAL, TEAL_BG)
        if i < 7:
            if i == 3:
                arrow(d, x + bw / 2, y + 51, x + bw / 2, y + 81, TEAL)
            elif r == 0:
                arrow(d, x + bw + 3, y + 24, x + bw + gap - 3, y + 24, TEAL)
            else:
                arrow(d, x - 3, y + 24, x - gap + 3, y + 24, TEAL)
    text(d, (20, 222), t["cmd_title"], 13, True, MUTED)
    code(d, 20, 236, W - 40, "bash training/setup.sh")
    code(d, 20, 268, W - 40, 'nohup bash training/run_train.sh "recognise skin cancer" > /dev/null 2>&1 &')
    code(d, 20, 300, W - 40, "tail -f logs/train_*.log")
    im.save(out)


def server(t, out):
    im, d = canvas(170)
    colors = [(TEAL, TEAL_BG), (BLUE, BLUE_BG), (VIOLET, VIOLET_BG), (AMBER, AMBER_BG)]
    bw, gap = 145, 60  # 4 x 145 + 3 x 60 = 760 = W - 40
    for i, (title, sub) in enumerate(zip(t["server_boxes"], t["server_sub"])):
        x = 20 + i * (bw + gap)
        fg, bg = colors[i]
        panel(d, x, 40, bw, 100, title, fg, bg, 13.5)
        text(d, (x + bw / 2, 104), sub, 11, fill=MUTED, anchor="mm")
        if i < 3:
            arrow(d, x + bw + 3, 90, x + bw + gap - 3, 90, both=(i == 2), label=t["server_links"][i])
    im.save(out)


def lan(t, out):
    im, d = canvas(360)
    panel(d, 20, 20, 210, 150, t["lan_pc"], BLUE, BLUE_BG)
    items(d, 36, 74, t["lan_pc_items"], BLUE, step=30)
    panel(d, W - 230, 20, 210, 150, t["lan_phone"], AMBER, AMBER_BG)
    items(d, W - 214, 74, t["lan_phone_items"], AMBER, step=30)
    text(d, (W / 2, 34), t["lan_net"], 12, True, MUTED, "mm")
    for i, s in enumerate(t["lan_nets"]):
        chip(d, 270, 52 + i * 40, 260, 32, s, VIOLET, VIOLET_BG, 12)
    arrow(d, 234, 95, 266, 95, both=True)
    arrow(d, 534, 95, W - 234, 95, both=True)
    y = 196
    cmds = ["pwsh lan/prepare_offline.ps1", "pwsh lan/start_lan.ps1 -OpenFirewall", "pwsh lan/start_lan.ps1"]
    for step, cmd in zip(t["lan_steps"], cmds):
        text(d, (20, y + 13), step, 12.5, True, INK)
        code(d, 250, y, W - 270, cmd)
        y += 40
    text(d, (20, y + 16), "-Https : local HTTPS certificate (install as an app)   ·   -Gpu : DirectML GPU acceleration", 11, fill=MUTED)
    im.save(out)


def where(t, out):
    im, d = canvas(178)
    text(d, (W / 2, 16), t["where_title"], 15, True, INK, "mm")
    colors = [(TEAL, TEAL_BG), (BLUE, BLUE_BG), (VIOLET, VIOLET_BG)]
    bw = (W - 40 - 2 * 20) / 3
    for i, (title, files, purpose) in enumerate(t["where"]):
        x = 20 + i * (bw + 20)
        fg, bg = colors[i]
        panel(d, x, 36, bw, 132, title, fg, bg, 14)
        y = 84
        for f in files:
            d.text(((x + 16) * S, y * S), f, font=ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", int(12 * S)),
                   fill=INK, anchor="lm")
            y += 22
        text(d, (x + bw / 2, 152), purpose, 12, True, fg, "mm")
    im.save(out)


def screens(t, out):
    """The two real app screenshots side by side in simple phone frames."""
    cw, pw, gap = 1200, 420, 40  # canvas width, phone screen width, frame padding (pixels)
    shots = []
    for name in ("app_start.png", "app_result.png"):
        im = Image.open(os.path.join(IMAGES, name))
        shots.append(im.resize((pw, int(im.height * pw / im.width)), Image.LANCZOS))
    top = 80
    h = top + max(p.height for p in shots) + 2 * 18 + 20
    im = Image.new("RGB", (cw, h), "white")
    d = ImageDraw.Draw(im)
    label_font = ImageFont.truetype(FONTS["bold"], 30)
    centers = [cw * 0.27, cw * 0.73]
    for cx, p, label in zip(centers, shots, t["screens"]):
        x = int(cx - pw / 2)
        d.rounded_rectangle([x - 18, top - 18, x + pw + 18, top + p.height + 18], radius=40, fill=(30, 41, 59))
        im.paste(p, (x, top))
        d.text((cx, top / 2 - 4), label, font=label_font, fill=INK, anchor="mm")
    ay = top + shots[0].height / 2
    d.polygon([(cw / 2 - 18, ay - 26), (cw / 2 + 22, ay), (cw / 2 - 18, ay + 26)], fill=GREY)
    im.save(out)


def main():
    for lang, t in T.items():
        folder = os.path.join(IMAGES, lang)
        os.makedirs(folder, exist_ok=True)
        overview(t, os.path.join(folder, "overview.png"))
        training(t, os.path.join(folder, "training.png"))
        server(t, os.path.join(folder, "server.png"))
        lan(t, os.path.join(folder, "lan.png"))
        where(t, os.path.join(folder, "where.png"))
        screens(t, os.path.join(folder, "app.png"))
    print("diagrams ok:", ", ".join(sorted(os.listdir(os.path.join(IMAGES, "en")))))


if __name__ == "__main__":
    main()
