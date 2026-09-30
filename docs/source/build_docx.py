"""Build the Word manual (.docx) from manual.html by writing Office Open XML directly.

Supported HTML: h1-h3, p (classes cover-title / cover-sub / cover-meta / caption / note), ul / ol / li,
table / tr / th / td (width:NN% on the first row), pre, b, code, sup, br page breaks, and the
placeholders [[TOC]], [[ARCH]], [[FLOWS]].
"""
import os
import re
import sys
import zipfile
from datetime import datetime, timezone
from html.parser import HTMLParser
from xml.sax.saxutils import escape

from PIL import Image

SP = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SP, "manual.docx")
PAGE_W, PAGE_H, MARGIN_LR, MARGIN_TB = 11906, 16838, 1418, 1440  # A4, dxa
TEXT_W = PAGE_W - 2 * MARGIN_LR
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")  # docs/images
IMG_PATTERN = re.compile(r"^\[\[IMG:([^|\]]+)(?:\|(\d+))?\]\]$")  # [[IMG:en/overview.png|80]] (width in % of text)
FONT_CN, FONT_CODE = "Calibri", "Consolas"
TEAL = "0F766E"


# ---------------------------------------------------------------- HTML -> blocks
class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.stack, self.runs, self.fmt = [], [], None, []
        self.list_stack, self.table, self.row, self.cell, self.in_body = [], None, None, None, False

    def _start_runs(self):
        self.runs = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "body":
            self.in_body = True
        if not self.in_body:
            return
        if tag in ("h1", "h2", "h3", "p", "li", "pre"):
            self.cur = {"tag": tag, "cls": a.get("class", ""), "style": a.get("style", "")}
            self._start_runs()
        elif tag in ("ul", "ol"):
            self.list_stack.append({"type": tag, "id": None})
        elif tag == "table":
            self.table = {"rows": []}
        elif tag == "tr":
            self.row = []
        elif tag in ("th", "td"):
            self.cell = {"header": tag == "th", "style": a.get("style", "")}
            self._start_runs()
        elif tag in ("b", "code", "sup"):
            self.fmt.append(tag)
        elif tag == "br" and "page-break" in a.get("style", ""):
            self.blocks.append({"type": "pagebreak"})

    def handle_endtag(self, tag):
        if not self.in_body:
            return
        if tag in ("h1", "h2", "h3", "p", "pre"):
            self.blocks.append({"type": tag, "cls": self.cur["cls"], "runs": self.runs})
            self.runs = None
        elif tag == "li":
            lst = self.list_stack[-1]
            self.blocks.append({"type": "li", "list": lst, "runs": self.runs})
            self.runs = None
        elif tag in ("ul", "ol"):
            self.list_stack.pop()
        elif tag in ("th", "td"):
            self.cell["runs"] = self.runs
            self.row.append(self.cell)
            self.runs = None
            self.cell = None
        elif tag == "tr":
            self.table["rows"].append(self.row)
        elif tag == "table":
            self.blocks.append({"type": "table", **self.table})
            self.table = None
        elif tag in ("b", "code", "sup") and self.fmt:
            self.fmt.pop()
        elif tag == "body":
            self.in_body = False

    def handle_data(self, data):
        if self.runs is None:
            return
        in_pre = self.cell is None and getattr(self, "cur", {}).get("tag") == "pre"
        if not in_pre:
            data = re.sub(r"\s+", " ", data)
        if data:
            self.runs.append((data, tuple(self.fmt)))


# ---------------------------------------------------------------- blocks -> WordprocessingML
def run_xml(text, fmt=(), extra=""):
    props = [extra] if extra else []
    if "b" in fmt:
        props.append("<w:b/><w:bCs/>")
    if "code" in fmt:
        props.append(f'<w:rFonts w:ascii="{FONT_CODE}" w:hAnsi="{FONT_CODE}" w:eastAsia="{FONT_CN}" w:cs="{FONT_CODE}"/>'
                     '<w:color w:val="0F5132"/><w:sz w:val="19"/>')
    if "sup" in fmt:
        props.append('<w:vertAlign w:val="superscript"/>')
    rpr = f"<w:rPr>{''.join(props)}</w:rPr>" if props else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def runs_xml(runs, extra=""):
    runs = list(runs or [])
    if runs:  # trim outer whitespace from the paragraph
        runs[0] = (runs[0][0].lstrip(), runs[0][1])
        runs[-1] = (runs[-1][0].rstrip(), runs[-1][1])
    return "".join(run_xml(t, f, extra) for t, f in runs if t)


def para(inner, style=None, ppr_extra=""):
    ppr = (f'<w:pStyle w:val="{style}"/>' if style else "") + ppr_extra
    return f"<w:p>{'<w:pPr>' + ppr + '</w:pPr>' if ppr else ''}{inner}</w:p>"


class Doc:
    def __init__(self):
        self.body, self.images, self.nums = [], [], []  # nums: list of ("bullet"|"decimal")
        self.bookmark = 0

    def image(self, rel_path, width_pct=100):
        path = os.path.join(IMG_DIR, rel_path)
        with Image.open(path) as im:
            w, h = im.size
        cx = int(TEXT_W * 635 * width_pct / 100)
        cy = int(cx * h / w)
        n = len(self.images) + 1
        rid, filename = f"rIdImg{n}", f"image{n}.png"
        self.images.append((rid, path, filename))
        drawing = (
            f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
            f'<wp:docPr id="{n}" name="Picture {n}"/><wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic>'
            f'<pic:nvPicPr><pic:cNvPr id="{n}" name="{filename}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r>')
        return para(drawing, ppr_extra='<w:jc w:val="center"/><w:keepNext/>')

    def toc(self):
        return (
            '<w:p><w:pPr><w:pStyle w:val="TOC1"/></w:pPr>'
            '<w:r><w:fldChar w:fldCharType="begin" w:dirty="true"/></w:r>'
            '<w:r><w:instrText xml:space="preserve"> TOC \\o "1-2" \\h \\z \\u </w:instrText></w:r>'
            '<w:r><w:fldChar w:fldCharType="separate"/></w:r>'
            + run_xml("(Update the table of contents when Word asks, or right-click here and choose Update Field.)")
            + '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p>')

    def table(self, rows):
        widths = []
        for c in rows[0]:
            m = re.search(r"width:\s*(\d+)%", c["style"])
            widths.append(int(m.group(1)) if m else 0)
        if not all(widths):
            widths = [100 // len(rows[0])] * len(rows[0])
        total = sum(widths)
        dxa = [round(TEXT_W * w / total) for w in widths]
        dxa[-1] = TEXT_W - sum(dxa[:-1])
        border = '<w:{s} w:val="single" w:sz="4" w:space="0" w:color="94A3B8"/>'
        borders = "".join(border.format(s=s) for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
        out = [f'<w:tbl><w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:tblBorders>{borders}</w:tblBorders>'
               '<w:tblLayout w:type="fixed"/><w:tblCellMar><w:top w:w="60" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
               '<w:bottom w:w="60" w:type="dxa"/><w:right w:w="100" w:type="dxa"/></w:tblCellMar></w:tblPr>'
               f'<w:tblGrid>{"".join(f"<w:gridCol w:w={chr(34)}{w}{chr(34)}/>" for w in dxa)}</w:tblGrid>']
        for ri, row in enumerate(rows):
            header = all(c["header"] for c in row)
            trpr = "<w:trPr><w:cantSplit/>" + ("<w:tblHeader/>" if header else "") + "</w:trPr>"
            cells = []
            for ci, c in enumerate(row):
                shade = f'<w:shd w:val="clear" w:color="auto" w:fill="{TEAL}"/>' if c["header"] else (
                    '<w:shd w:val="clear" w:color="auto" w:fill="F8FAFC"/>' if ri % 2 == 0 else "")
                extra = '<w:color w:val="FFFFFF"/><w:b/>' if c["header"] else ""
                cells.append(f'<w:tc><w:tcPr><w:tcW w:w="{dxa[ci]}" w:type="dxa"/>{shade}</w:tcPr>'
                             f'{para(runs_xml(c["runs"], extra), "TableText")}</w:tc>')
            out.append(f"<w:tr>{trpr}{''.join(cells)}</w:tr>")
        out.append("</w:tbl>")
        out.append(para("", "Spacer"))
        return "".join(out)

    def list_item(self, block):
        lst = block["list"]
        if lst["id"] is None:  # every list gets its own numbering instance (ordered lists restart at 1)
            self.nums.append("bullet" if lst["type"] == "ul" else "decimal")
            lst["id"] = len(self.nums)
        return para(runs_xml(block["runs"]), "ListParagraph",
                    f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{lst["id"]}"/></w:numPr>')

    def add(self, b):
        t = b["type"]
        if t == "pagebreak":
            self.body.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')
        elif t in ("h1", "h2", "h3"):
            self.bookmark += 1
            self.body.append(para(runs_xml(b["runs"]), {"h1": "Heading1", "h2": "Heading2", "h3": "Heading3"}[t]))
        elif t == "p":
            text = "".join(r[0] for r in b["runs"]).strip()
            if text == "[[TOC]]":
                self.body.append(self.toc())
            elif IMG_PATTERN.match(text):
                m = IMG_PATTERN.match(text)
                self.body.append(self.image(m.group(1), int(m.group(2) or 100)))
            elif text == "Contents":
                self.body.append(para(run_xml("Contents"), "TOCHeading"))
            else:
                style = {"cover-title": "CoverTitle", "cover-sub": "CoverSub", "cover-meta": "CoverMeta",
                         "caption": "Caption", "note": "Note"}.get(b["cls"], None)
                self.body.append(para(runs_xml(b["runs"]), style))
        elif t == "pre":
            text = "".join(r[0] for r in b["runs"]).strip("\n")
            for line in text.split("\n"):
                self.body.append(para(run_xml(line), "Code"))
            self.body.append(para("", "Spacer"))
        elif t == "li":
            self.body.append(self.list_item(b))
        elif t == "table":
            self.body.append(self.table(b["rows"]))


PPR_ORDER = ["pStyle", "keepNext", "keepLines", "pageBreakBefore", "framePr", "widowControl", "numPr",
             "suppressLineNumbers", "pBdr", "shd", "tabs", "suppressAutoHyphens", "kinsoku", "wordWrap",
             "overflowPunct", "topLinePunct", "autoSpaceDE", "autoSpaceDN", "bidi", "adjustRightInd", "snapToGrid",
             "spacing", "ind", "contextualSpacing", "mirrorIndents", "suppressOverlap", "jc", "textDirection",
             "textAlignment", "textboxTightWrap", "outlineLvl", "divId", "cnfStyle", "rPr", "sectPr", "pPrChange"]


def order_ppr(xml):
    """Sort the child elements of every <w:pPr> by the schema order (Word rejects other orders)."""
    def fix(m):
        items = re.findall(r"(<w:(\w+)(?:[^<>]*/>|[^<>]*>.*?</w:\2>))", m.group(1), flags=re.S)
        items.sort(key=lambda it: PPR_ORDER.index(it[1]))
        return "<w:pPr>" + "".join(it[0] for it in items) + "</w:pPr>"
    return re.sub(r"<w:pPr>(.*?)</w:pPr>", fix, xml, flags=re.S)


# ---------------------------------------------------------------- package parts
NS = ('xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
      'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"')


def styles_xml():
    def font(ascii_=FONT_CN, ea=FONT_CN):
        return f'<w:rFonts w:ascii="{ascii_}" w:hAnsi="{ascii_}" w:eastAsia="{ea}" w:cs="{ascii_}"/>'

    def pstyle(sid, name, ppr="", rpr="", based="Normal", nxt="Normal", extra=""):
        return (f'<w:style w:type="paragraph" w:styleId="{sid}"><w:name w:val="{name}"/><w:basedOn w:val="{based}"/>'
                f'<w:next w:val="{nxt}"/>{extra}<w:qFormat/><w:pPr>{ppr}</w:pPr><w:rPr>{rpr}</w:rPr></w:style>')

    heading = lambda lvl, size, color, before, after: pstyle(
        f"Heading{lvl}", f"heading {lvl}",
        f'<w:keepNext/><w:keepLines/><w:spacing w:before="{before}" w:after="{after}"/><w:outlineLvl w:val="{lvl - 1}"/>'
        + ('<w:pBdr><w:bottom w:val="single" w:sz="8" w:space="4" w:color="0F766E"/></w:pBdr>' if lvl == 1 else ""),
        f'<w:b/><w:bCs/><w:color w:val="{color}"/><w:sz w:val="{size}"/><w:szCs w:val="{size}"/>',
        extra='<w:uiPriority w:val="9"/>')
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:styles {NS}>'
            f'<w:docDefaults><w:rPrDefault><w:rPr>{font()}<w:sz w:val="22"/><w:szCs w:val="22"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>'
            '<w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>'
            '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/>'
            '<w:rPr><w:color w:val="0F172A"/></w:rPr></w:style>'
            + heading(1, 34, TEAL, 480, 200) + heading(2, 28, "115E59", 320, 120) + heading(3, 24, "1E293B", 240, 80)
            + pstyle("CoverTitle", "Cover Title", '<w:jc w:val="center"/><w:spacing w:before="3600" w:after="240"/>',
                     f'<w:b/><w:color w:val="{TEAL}"/><w:sz w:val="56"/><w:szCs w:val="56"/>')
            + pstyle("CoverSub", "Cover Subtitle", '<w:jc w:val="center"/><w:spacing w:after="1600"/>',
                     '<w:color w:val="334155"/><w:sz w:val="32"/><w:szCs w:val="32"/>')
            + pstyle("CoverMeta", "Cover Meta", '<w:jc w:val="center"/><w:spacing w:after="120"/>',
                     '<w:color w:val="475569"/><w:sz w:val="22"/>')
            + pstyle("Caption", "caption", '<w:jc w:val="center"/><w:spacing w:before="60" w:after="240"/>',
                     '<w:color w:val="475569"/><w:sz w:val="18"/>')
            + pstyle("Note", "Note", '<w:pBdr><w:left w:val="single" w:sz="24" w:space="8" w:color="B45309"/></w:pBdr>'
                     '<w:shd w:val="clear" w:color="auto" w:fill="FEF3C7"/><w:ind w:left="200" w:right="100"/><w:spacing w:before="120" w:after="200"/>')
            + pstyle("Code", "Code", '<w:shd w:val="clear" w:color="auto" w:fill="F1F5F9"/><w:spacing w:after="0" w:line="260" w:lineRule="auto"/>'
                     '<w:ind w:left="120" w:right="120"/>', font(FONT_CODE, FONT_CN) + '<w:sz w:val="18"/><w:szCs w:val="18"/>')
            + pstyle("TableText", "Table Text", '<w:spacing w:after="0" w:line="264" w:lineRule="auto"/>', '<w:sz w:val="19"/><w:szCs w:val="19"/>')
            + pstyle("Spacer", "Spacer", '<w:spacing w:after="120" w:line="120" w:lineRule="exact"/>', '<w:sz w:val="8"/>')
            + pstyle("ListParagraph", "List Paragraph", '<w:spacing w:after="60"/><w:ind w:left="420" w:hanging="300"/>')
            + pstyle("TOCHeading", "TOC Heading", '<w:spacing w:after="240"/>', f'<w:b/><w:color w:val="{TEAL}"/><w:sz w:val="34"/>')
            + pstyle("TOC1", "toc 1", '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9060"/></w:tabs><w:spacing w:before="40" w:after="0" w:line="230" w:lineRule="auto"/>', '<w:b/><w:sz w:val="19"/>')
            + pstyle("TOC2", "toc 2", '<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="9060"/></w:tabs><w:ind w:left="420"/><w:spacing w:after="0" w:line="222" w:lineRule="auto"/>', '<w:sz w:val="18"/>')
            + '<w:style w:type="character" w:styleId="Hyperlink"><w:name w:val="Hyperlink"/><w:rPr><w:color w:val="0F766E"/></w:rPr></w:style>'
            + '</w:styles>')


def numbering_xml(nums):
    abstract = (
        '<w:abstractNum w:abstractNumId="0"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/>'
        '<w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="420" w:hanging="300"/></w:pPr>'
        f'<w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial"/><w:color w:val="{TEAL}"/></w:rPr></w:lvl></w:abstractNum>'
        '<w:abstractNum w:abstractNumId="1"><w:multiLevelType w:val="singleLevel"/><w:lvl w:ilvl="0"><w:start w:val="1"/>'
        '<w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="420" w:hanging="360"/></w:pPr>'
        f'<w:rPr><w:b/><w:color w:val="{TEAL}"/></w:rPr></w:lvl></w:abstractNum>')
    inst = "".join(f'<w:num w:numId="{i}"><w:abstractNumId w:val="{0 if k == "bullet" else 1}"/>'
                   + ('<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride>' if k == "decimal" else "")
                   + "</w:num>" for i, k in enumerate(nums, 1))
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:numbering {NS}>{abstract}{inst}</w:numbering>'


def footer_xml():
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:ftr {NS}><w:p><w:pPr><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:rPr><w:color w:val="64748B"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="begin"/></w:r>'
            '<w:r><w:rPr><w:color w:val="64748B"/><w:sz w:val="18"/></w:rPr><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
            '<w:r><w:rPr><w:color w:val="64748B"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="separate"/></w:r>'
            '<w:r><w:rPr><w:color w:val="64748B"/><w:sz w:val="18"/></w:rPr><w:t>1</w:t></w:r>'
            '<w:r><w:rPr><w:color w:val="64748B"/><w:sz w:val="18"/></w:rPr><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>')


def build():
    p = Parser()
    p.feed(open(os.path.join(SP, "manual.html"), encoding="utf-8").read())
    d = Doc()
    for b in p.blocks:
        d.add(b)
    sect = (f'<w:sectPr><w:footerReference w:type="default" r:id="rIdFooter"/><w:pgSz w:w="{PAGE_W}" w:h="{PAGE_H}"/>'
            f'<w:pgMar w:top="{MARGIN_TB}" w:right="{MARGIN_LR}" w:bottom="{MARGIN_TB}" w:left="{MARGIN_LR}" w:header="709" w:footer="709" w:gutter="0"/>'
            '<w:titlePg/></w:sectPr>')
    document = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document {NS}><w:body>'
                + "".join(d.body) + sect + "</w:body></w:document>")
    rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '<Relationship Id="rIdNumbering" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>'
            '<Relationship Id="rIdSettings" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>'
            '<Relationship Id="rIdFooter" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/>'
            + "".join(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{fn}"/>'
                      for rid, _, fn in d.images) + "</Relationships>")
    settings = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:settings {NS}>'
                '<w:defaultTabStop w:val="420"/><w:updateFields w:val="true"/>'
                '<w:compat><w:compatSetting w:name="compatibilityMode" w:uri="http://schemas.microsoft.com/office/word" w:val="15"/></w:compat></w:settings>')
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
            'xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
            '<dc:title>Cancer Image Diagnosis System - Architecture and User Guide</dc:title><dc:language>en-US</dc:language>'
            f'<dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created><dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>'
            '</cp:coreProperties>')
    content_types = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                     '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                     '<Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/>'
                     '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                     '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                     '<Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>'
                     '<Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>'
                     '<Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/>'
                     '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/></Types>')
    root_rels = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                 '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                 '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
                 '</Relationships>')
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("word/document.xml", order_ppr(document))
        z.writestr("word/styles.xml", order_ppr(styles_xml()))
        z.writestr("word/numbering.xml", order_ppr(numbering_xml(d.nums)))
        z.writestr("word/settings.xml", settings)
        z.writestr("word/footer1.xml", footer_xml())
        z.writestr("word/_rels/document.xml.rels", rels)
        for _, path, fn in d.images:
            z.write(path, f"word/media/{fn}")
    counts = {k: sum(1 for b in p.blocks if b["type"] == k) for k in ("h1", "h2", "table", "li", "pre")}
    print(f"wrote {OUT} | {counts} | images {len(d.images)} | lists {len(d.nums)}")


if __name__ == "__main__":
    build()
