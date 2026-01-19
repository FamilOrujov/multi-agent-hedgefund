
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import io
import re

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether, Flowable
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.widgets.markers import makeMarker

from src.config.settings import get_settings

PROJECT_NAME = "Aegis Flux"
PROJECT_TAGLINE = "Multi-Agent AI Investment Analysis"


COLORS = {
    "primary": colors.HexColor("#00d4ff"),
    "secondary": colors.HexColor("#8b5cf6"),
    "professional_blue": colors.HexColor("#1e3a5f"),
    "success": colors.HexColor("#10b981"),
    "warning": colors.HexColor("#f59e0b"),
    "danger": colors.HexColor("#ef4444"),
    "dark": colors.HexColor("#1f2937"),
    "light": colors.HexColor("#f3f4f6"),
    "muted": colors.HexColor("#6b7280"),
    "white": colors.white,
    "black": colors.HexColor("#111827"),
}

DECISION_COLORS = {
    "BUY": COLORS["success"],
    "SELL": COLORS["danger"],
    "HOLD": COLORS["warning"],
}


class PDFReportGenerator:
    """
    Professional PDF report generator for hedge fund analysis.
    
    Creates detailed, modern reports with:
    - Executive summary
    - Technical analysis with charts
    - Fundamental analysis
    - Sentiment analysis
    - Risk assessment
    - Investment thesis
    """

    def __init__(self, output_dir: Optional[str] = None):
        settings = get_settings()
        self.output_dir = Path(output_dir or settings.data_dir / "reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.styles = self._create_styles()

    def _create_styles(self) -> dict:
        """Create custom paragraph styles."""
        base_styles = getSampleStyleSheet()
        
        styles = {
            "title": ParagraphStyle(
                "CustomTitle",
                parent=base_styles["Title"],
                fontSize=28,
                textColor=COLORS["dark"],
                spaceAfter=6,
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "subtitle": ParagraphStyle(
                "CustomSubtitle",
                parent=base_styles["Normal"],
                fontSize=14,
                textColor=COLORS["muted"],
                spaceAfter=20,
                alignment=TA_CENTER,
            ),
            "heading1": ParagraphStyle(
                "CustomH1",
                parent=base_styles["Heading1"],
                fontSize=18,
                textColor=COLORS["dark"],
                spaceBefore=20,
                spaceAfter=12,
                fontName="Helvetica-Bold",
                borderColor=COLORS["primary"],
                borderWidth=0,
                borderPadding=0,
            ),
            "heading2": ParagraphStyle(
                "CustomH2",
                parent=base_styles["Heading2"],
                fontSize=14,
                textColor=COLORS["professional_blue"],
                spaceBefore=15,
                spaceAfter=8,
                fontName="Helvetica-Bold",
            ),
            "body": ParagraphStyle(
                "CustomBody",
                parent=base_styles["Normal"],
                fontSize=10,
                textColor=COLORS["dark"],
                spaceAfter=8,
                alignment=TA_JUSTIFY,
                leading=14,
            ),
            "body_small": ParagraphStyle(
                "CustomBodySmall",
                parent=base_styles["Normal"],
                fontSize=9,
                textColor=COLORS["muted"],
                spaceAfter=6,
            ),
            "decision_buy": ParagraphStyle(
                "DecisionBuy",
                parent=base_styles["Normal"],
                fontSize=24,
                textColor=COLORS["success"],
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "decision_sell": ParagraphStyle(
                "DecisionSell",
                parent=base_styles["Normal"],
                fontSize=24,
                textColor=COLORS["danger"],
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "decision_hold": ParagraphStyle(
                "DecisionHold",
                parent=base_styles["Normal"],
                fontSize=24,
                textColor=COLORS["warning"],
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "metric_value": ParagraphStyle(
                "MetricValue",
                parent=base_styles["Normal"],
                fontSize=16,
                textColor=COLORS["primary"],
                alignment=TA_CENTER,
                fontName="Helvetica-Bold",
            ),
            "metric_label": ParagraphStyle(
                "MetricLabel",
                parent=base_styles["Normal"],
                fontSize=9,
                textColor=COLORS["muted"],
                alignment=TA_CENTER,
            ),
        }
        
        return styles

    def _parse_markdown(self, text: str) -> list:
        """
        Parse markdown text and convert to ReportLab flowables.
        Handles: **bold**, headers, numbered lists, bullet points.
        """
        if not text:
            return []
        
        # Handle dict input (error case)
        if isinstance(text, dict):
            if "error" in text:
                return [Paragraph(f"Error: {text['error']}", self.styles["body"])]
            text = str(text)
        
        elements = []
        lines = text.split('\n')
        current_paragraph = []
        in_list = False
        
        for line in lines:
            line = line.strip()
            
            if not line:
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    para_text = self._convert_inline_markdown(para_text)
                    elements.append(Paragraph(para_text, self.styles["body"]))
                    current_paragraph = []
                in_list = False
                continue
            
            # Headers
            if line.startswith('###'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    para_text = self._convert_inline_markdown(para_text)
                    elements.append(Paragraph(para_text, self.styles["body"]))
                    current_paragraph = []
                header_text = line.lstrip('#').strip()
                header_text = self._convert_inline_markdown(header_text)
                elements.append(Spacer(1, 8))
                elements.append(Paragraph(header_text, self.styles["heading2"]))
                continue
            elif line.startswith('##'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    para_text = self._convert_inline_markdown(para_text)
                    elements.append(Paragraph(para_text, self.styles["body"]))
                    current_paragraph = []
                header_text = line.lstrip('#').strip()
                header_text = self._convert_inline_markdown(header_text)
                elements.append(Spacer(1, 10))
                elements.append(Paragraph(header_text, self.styles["heading2"]))
                continue
            elif line.startswith('#'):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    para_text = self._convert_inline_markdown(para_text)
                    elements.append(Paragraph(para_text, self.styles["body"]))
                    current_paragraph = []
                header_text = line.lstrip('#').strip()
                header_text = self._convert_inline_markdown(header_text)
                elements.append(Spacer(1, 12))
                elements.append(Paragraph(header_text, self.styles["heading1"]))
                continue
            
            # Numbered lists (1. 2. etc)
            numbered_match = re.match(r'^(\d+)\.\s+(.+)$', line)
            if numbered_match:
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    para_text = self._convert_inline_markdown(para_text)
                    elements.append(Paragraph(para_text, self.styles["body"]))
                    current_paragraph = []
                num = numbered_match.group(1)
                item_text = numbered_match.group(2)
                item_text = self._convert_inline_markdown(item_text)
                elements.append(Paragraph(
                    f"<b>{num}.</b> {item_text}",
                    self.styles["body"]
                ))
                in_list = True
                continue
            
            # Bullet points
            if line.startswith('- ') or line.startswith('* ') or line.startswith('• '):
                if current_paragraph:
                    para_text = ' '.join(current_paragraph)
                    para_text = self._convert_inline_markdown(para_text)
                    elements.append(Paragraph(para_text, self.styles["body"]))
                    current_paragraph = []
                item_text = line[2:].strip()
                item_text = self._convert_inline_markdown(item_text)
                elements.append(Paragraph(
                    f"• {item_text}",
                    self.styles["body"]
                ))
                in_list = True
                continue
            
            # Regular text - accumulate into paragraph
            current_paragraph.append(line)
        
        # Handle remaining paragraph
        if current_paragraph:
            para_text = ' '.join(current_paragraph)
            para_text = self._convert_inline_markdown(para_text)
            elements.append(Paragraph(para_text, self.styles["body"]))
        
        return elements

    def _convert_inline_markdown(self, text: str) -> str:
        """Convert inline markdown (bold, italic) to ReportLab XML tags."""
        # Escape XML special characters first
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Convert **bold** to <b>bold</b>
        text = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', text)
        
        # Convert *italic* to <i>italic</i> (but not if already processed as bold)
        text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
        
        # Convert `code` to <font face="Courier">code</font>
        text = re.sub(r'`([^`]+)`', r'<font face="Courier" size="9">\1</font>', text)
        
        return text

    def generate(
        self,
        result: dict[str, Any],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Generate a comprehensive PDF report from analysis results.
        
        Args:
            result: Analysis result dictionary from the workflow
            filename: Optional custom filename
            
        Returns:
            Path to the generated PDF file
        """
        ticker = result.get("ticker", "UNKNOWN")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if filename is None:
            filename = f"{ticker}_analysis_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        doc = SimpleDocTemplate(
            str(filepath),
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch,
        )
        
        story = []
        
        story.extend(self._create_cover_page(result))
        story.append(PageBreak())
        
        story.extend(self._create_executive_summary(result))
        story.append(PageBreak())
        
        story.extend(self._create_technical_section(result))
        story.append(PageBreak())
        
        story.extend(self._create_fundamental_section(result))
        story.append(PageBreak())
        
        story.extend(self._create_sentiment_section(result))
        story.append(PageBreak())
        
        story.extend(self._create_thesis_section(result))
        
        story.extend(self._create_disclaimer())
        
        doc.build(story, onFirstPage=self._add_header_footer, 
                  onLaterPages=self._add_header_footer)
        
        return filepath

    def _create_cover_page(self, result: dict[str, Any]) -> list:
        """Create the cover page."""
        elements = []
        
        elements.append(Spacer(1, 1.5*inch))
        
        elements.append(Paragraph(PROJECT_NAME.upper(), self.styles["title"]))
        elements.append(Paragraph(PROJECT_TAGLINE, self.styles["subtitle"]))
        
        elements.append(Spacer(1, 0.5*inch))
        
        elements.append(HRFlowable(
            width="80%", thickness=2, color=COLORS["primary"],
            spaceBefore=10, spaceAfter=30, hAlign="CENTER"
        ))
        
        ticker = result.get("ticker", "N/A")
        company_info = result.get("company_info", {})
        company_name = company_info.get("name", ticker)
        
        # Only show ticker symbol here
        elements.append(Paragraph(ticker, ParagraphStyle(
            "Ticker", fontSize=48, textColor=COLORS["dark"],
            alignment=TA_CENTER, fontName="Helvetica-Bold",
            spaceAfter=20
        )))
        
        elements.append(Spacer(1, 0.3*inch))
        
        decision = result.get("manager_decision", {})
        decision_text = decision.get("decision", "HOLD")
        confidence = decision.get("confidence", 0)
        
        decision_style = self.styles[f"decision_{decision_text.lower()}"]
        elements.append(Paragraph(f"▶ {decision_text} ◀", decision_style))
        
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(
            f"Confidence: {confidence}%",
            ParagraphStyle("Confidence", fontSize=14, textColor=COLORS["muted"],
                          alignment=TA_CENTER)
        ))
        
        # Company name below Confidence
        elements.append(Spacer(1, 0.4*inch))
        elements.append(Paragraph(company_name, ParagraphStyle(
            "CompanyName", fontSize=16, textColor=COLORS["dark"],
            alignment=TA_CENTER, fontName="Helvetica"
        )))
        
        # Generated date - will be shown above page number in footer
        # Store date for footer use
        self._cover_date = datetime.now().strftime("%B %d, %Y")
        
        return elements

    def _create_signal_chart(self, tech_signal: dict) -> Drawing:
        """Create a pie chart showing bullish vs bearish signal distribution."""
        bullish = tech_signal.get("bullish_count", 0)
        bearish = tech_signal.get("bearish_count", 0)
        total = tech_signal.get("total_indicators", 0)
        neutral = max(0, total - bullish - bearish)
        
        if total == 0:
            return None
        
        drawing = Drawing(200, 120)
        
        pie = Pie()
        pie.x = 60
        pie.y = 10
        pie.width = 80
        pie.height = 80
        
        data = []
        labels = []
        slice_colors = []
        
        if bullish > 0:
            data.append(bullish)
            labels.append(f"Bullish ({bullish})")
            slice_colors.append(COLORS["success"])
        if bearish > 0:
            data.append(bearish)
            labels.append(f"Bearish ({bearish})")
            slice_colors.append(COLORS["danger"])
        if neutral > 0:
            data.append(neutral)
            labels.append(f"Neutral ({neutral})")
            slice_colors.append(COLORS["warning"])
        
        if not data:
            return None
            
        pie.data = data
        pie.labels = labels
        
        for i, color in enumerate(slice_colors):
            pie.slices[i].fillColor = color
            pie.slices[i].strokeColor = colors.white
            pie.slices[i].strokeWidth = 1
        
        pie.sideLabels = True
        pie.slices.fontName = "Helvetica"
        pie.slices.fontSize = 8
        
        drawing.add(pie)
        
        # Add title
        drawing.add(String(100, 105, "Signal Distribution", 
                          fontName="Helvetica-Bold", fontSize=9, 
                          textAnchor="middle"))
        
        return drawing

    def _create_executive_summary(self, result: dict[str, Any]) -> list:
        """Create executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles["heading1"]))
        elements.append(HRFlowable(
            width="100%", thickness=1, color=COLORS["primary"],
            spaceBefore=0, spaceAfter=15
        ))
        
        decision = result.get("manager_decision", {})
        
        # Decision metrics table
        metrics_data = [
            ["Decision", "Confidence", "Position Size", "Consensus"],
            [
                decision.get("decision", "HOLD"),
                f"{decision.get('confidence', 0)}%",
                decision.get("position_size", "None"),
                "Yes" if decision.get("has_consensus") else "No"
            ]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[1.5*inch]*4)
        
        # Color the decision cell based on decision type
        decision_text = decision.get("decision", "HOLD")
        decision_color = COLORS["success"] if decision_text == "BUY" else (
            COLORS["danger"] if decision_text == "SELL" else COLORS["warning"]
        )
        
        metrics_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["dark"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("BACKGROUND", (0, 1), (0, 1), decision_color),
            ("TEXTCOLOR", (0, 1), (0, 1), COLORS["white"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 10),
            ("FONTSIZE", (0, 1), (-1, -1), 12),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("TOPPADDING", (0, 1), (-1, -1), 15),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 15),
            ("GRID", (0, 0), (-1, -1), 1, COLORS["light"]),
            ("BOX", (0, 0), (-1, -1), 2, COLORS["primary"]),
        ]))
        
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
        
        # Agent Signals section with chart
        elements.append(Paragraph("Agent Signals", self.styles["heading2"]))
        
        signals_data = [["Agent", "Signal", "Confidence"]]
        
        for agent, key in [
            ("Technical Analyst", "technical_signal"),
            ("Fundamental Analyst", "fundamental_signal"),
            ("Sentiment Analyst", "sentiment_signal"),
        ]:
            signal = result.get(key, {})
            signal_text = signal.get("signal", "N/A")
            conf = signal.get("confidence", 0)
            conf_str = f"{conf:.0%}" if isinstance(conf, float) else f"{conf}%"
            signals_data.append([agent, signal_text, conf_str])
        
        signals_table = Table(signals_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
        signals_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLORS["secondary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("TOPPADDING", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
            ("BOX", (0, 0), (-1, -1), 1, COLORS["secondary"]),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light"]]),
        ]))
        
        elements.append(signals_table)
        elements.append(Spacer(1, 15))
        
        # Add signal distribution chart if technical signal has counts
        tech_signal = result.get("technical_signal", {})
        if tech_signal.get("total_indicators", 0) > 0:
            chart = self._create_signal_chart(tech_signal)
            if chart:
                elements.append(Paragraph("Technical Indicator Distribution", self.styles["heading2"]))
                elements.append(chart)
        
        # Key Highlights section
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("Key Highlights", self.styles["heading2"]))
        
        highlights = []
        
        # Technical highlight
        tech_conf = tech_signal.get("confidence", 0)
        if isinstance(tech_conf, float):
            tech_conf = int(tech_conf * 100)
        tech_sig = tech_signal.get("signal", "Neutral")
        highlights.append(f"<b>Technical:</b> {tech_sig} signal with {tech_conf}% confidence")
        
        # Fundamental highlight
        fund_signal = result.get("fundamental_signal", {})
        fund_conf = fund_signal.get("confidence", 0)
        if isinstance(fund_conf, float):
            fund_conf = int(fund_conf * 100)
        fund_sig = fund_signal.get("signal", "Neutral")
        highlights.append(f"<b>Fundamental:</b> {fund_sig} outlook with {fund_conf}% confidence")
        
        # Sentiment highlight
        sent_signal = result.get("sentiment_signal", {})
        sent_score = result.get("sentiment_score", 0)
        sent_sig = sent_signal.get("signal", "Neutral")
        highlights.append(f"<b>Sentiment:</b> {sent_sig} with score of {sent_score:.2f}")
        
        for highlight in highlights:
            elements.append(Paragraph(f"• {highlight}", self.styles["body"]))
        
        return elements

    def _create_technical_section(self, result: dict[str, Any]) -> list:
        """Create technical analysis section."""
        elements = []
        
        elements.append(Paragraph("Technical Analysis", self.styles["heading1"]))
        elements.append(HRFlowable(
            width="100%", thickness=1, color=COLORS["primary"],
            spaceBefore=0, spaceAfter=15
        ))
        
        indicators = result.get("technical_indicators", {})
        tech_signal = result.get("technical_signal", {})
        
        # Signal summary box
        if tech_signal:
            signal_text = tech_signal.get("signal", "Neutral")
            confidence = tech_signal.get("confidence", 0)
            bullish = tech_signal.get("bullish_count", 0)
            bearish = tech_signal.get("bearish_count", 0)
            total = tech_signal.get("total_indicators", 0)
            
            signal_color = COLORS["success"] if signal_text == "Bullish" else (
                COLORS["danger"] if signal_text == "Bearish" else COLORS["warning"]
            )
            
            summary_data = [
                ["Overall Signal", "Confidence", "Bullish", "Bearish", "Total"],
                [signal_text, f"{confidence:.0%}", str(bullish), str(bearish), str(total)]
            ]
            summary_table = Table(summary_data, colWidths=[1.3*inch]*5)
            summary_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), COLORS["dark"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
                ("BACKGROUND", (0, 1), (0, 1), signal_color),
                ("TEXTCOLOR", (0, 1), (0, 1), COLORS["white"]),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 1, COLORS["light"]),
                ("BOX", (0, 0), (-1, -1), 2, COLORS["primary"]),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]))
            elements.append(summary_table)
            elements.append(Spacer(1, 15))
        
        if indicators:
            elements.append(Paragraph("Momentum Indicators", self.styles["heading2"]))
            
            momentum_data = [["Indicator", "Value", "Signal"]]
            
            if "rsi" in indicators and "value" in indicators["rsi"]:
                rsi = indicators["rsi"]
                momentum_data.append([
                    "RSI (14)",
                    f"{rsi.get('value', 0):.1f}",
                    rsi.get("signal", "N/A")
                ])
            
            if "stochastic" in indicators and "k" in indicators["stochastic"]:
                stoch = indicators["stochastic"]
                momentum_data.append([
                    "Stochastic %K/%D",
                    f"{stoch.get('k', 0):.1f} / {stoch.get('d', 0):.1f}",
                    stoch.get("signal", "N/A")
                ])
            
            if "williams_r" in indicators and "value" in indicators["williams_r"]:
                wr = indicators["williams_r"]
                momentum_data.append([
                    "Williams %R",
                    f"{wr.get('value', 0):.1f}",
                    wr.get("signal", "N/A")
                ])
            
            if "cci" in indicators and "value" in indicators["cci"]:
                cci = indicators["cci"]
                momentum_data.append([
                    "CCI (20)",
                    f"{cci.get('value', 0):.1f}",
                    cci.get("signal", "N/A")
                ])
            
            if "roc" in indicators and "value" in indicators["roc"]:
                roc = indicators["roc"]
                momentum_data.append([
                    "ROC (12)",
                    f"{roc.get('value', 0):.2f}%",
                    roc.get("signal", "N/A")
                ])
            
            if len(momentum_data) > 1:
                mom_table = Table(momentum_data, colWidths=[2*inch, 2*inch, 2*inch])
                mom_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLORS["secondary"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light"]]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(mom_table)
                elements.append(Spacer(1, 10))
                
                # Add detailed explanations for momentum indicators
                elements.append(Paragraph("<b>Momentum Indicator Interpretations:</b>", self.styles["body"]))
                
                if "rsi" in indicators and "value" in indicators["rsi"]:
                    rsi_val = indicators["rsi"].get("value", 0)
                    rsi_interp = ""
                    if rsi_val >= 70:
                        rsi_interp = "The RSI is in overbought territory (above 70), suggesting the stock may be overextended and due for a pullback. This is typically a bearish signal for short-term traders."
                    elif rsi_val <= 30:
                        rsi_interp = "The RSI is in oversold territory (below 30), suggesting the stock may be undervalued and due for a bounce. This is typically a bullish signal for contrarian traders."
                    elif rsi_val >= 50:
                        rsi_interp = f"The RSI at {rsi_val:.1f} indicates bullish momentum as it's above the 50 midline. The stock is showing positive price momentum."
                    else:
                        rsi_interp = f"The RSI at {rsi_val:.1f} indicates bearish momentum as it's below the 50 midline. The stock is showing negative price momentum."
                    elements.append(Paragraph(f"• <b>RSI ({rsi_val:.1f}):</b> {rsi_interp}", self.styles["body"]))
                
                if "stochastic" in indicators and "k" in indicators["stochastic"]:
                    stoch = indicators["stochastic"]
                    k_val = stoch.get("k", 0)
                    d_val = stoch.get("d", 0)
                    if k_val > 80:
                        stoch_interp = f"Stochastic %K at {k_val:.1f} is in overbought territory. When %K crosses below %D from this level, it generates a sell signal."
                    elif k_val < 20:
                        stoch_interp = f"Stochastic %K at {k_val:.1f} is in oversold territory. When %K crosses above %D from this level, it generates a buy signal."
                    elif k_val > d_val:
                        stoch_interp = f"Stochastic shows a bullish crossover with %K ({k_val:.1f}) above %D ({d_val:.1f}), indicating upward momentum."
                    else:
                        stoch_interp = f"Stochastic shows a bearish crossover with %K ({k_val:.1f}) below %D ({d_val:.1f}), indicating downward momentum."
                    elements.append(Paragraph(f"• <b>Stochastic:</b> {stoch_interp}", self.styles["body"]))
                
                elements.append(Spacer(1, 10))
            
            # Trend Indicators section
            elements.append(Paragraph("Trend Indicators", self.styles["heading2"]))
            trend_data = [["Indicator", "Value", "Signal"]]
            
            if "macd" in indicators and "histogram" in indicators["macd"]:
                macd = indicators["macd"]
                trend_data.append([
                    "MACD",
                    f"Hist: {macd.get('histogram', 0):.2f}",
                    macd.get("signal", "N/A")
                ])
            
            if "adx" in indicators and "value" in indicators["adx"]:
                adx = indicators["adx"]
                trend_data.append([
                    "ADX (14)",
                    f"{adx.get('value', 0):.1f}",
                    f"{adx.get('trend_strength', 'N/A')} - {adx.get('signal', 'N/A')}"
                ])
            
            if "moving_averages" in indicators and "sma_20" in indicators["moving_averages"]:
                ma = indicators["moving_averages"]
                trend_data.append([
                    "SMA 20/50",
                    f"{ma.get('sma_20', 0):.2f} / {ma.get('sma_50', 0):.2f}",
                    ma.get("trend", "N/A")
                ])
            
            if len(trend_data) > 1:
                trend_table = Table(trend_data, colWidths=[2*inch, 2*inch, 2*inch])
                trend_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLORS["primary"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light"]]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(trend_table)
                elements.append(Spacer(1, 10))
            
            # Volatility & Volume
            elements.append(Paragraph("Volatility & Volume", self.styles["heading2"]))
            vol_data = [["Indicator", "Value", "Interpretation"]]
            
            if "bollinger_bands" in indicators and "current_price" in indicators["bollinger_bands"]:
                bb = indicators["bollinger_bands"]
                vol_data.append([
                    "Bollinger Bands",
                    f"Price: {bb.get('current_price', 0):.2f}",
                    bb.get("signal", "N/A")
                ])
            
            if "atr" in indicators and "value" in indicators["atr"]:
                atr = indicators["atr"]
                vol_data.append([
                    "ATR (14)",
                    f"{atr.get('value', 0):.2f}",
                    atr.get("volatility", "N/A")
                ])
            
            if "obv" in indicators and "trend" in indicators["obv"]:
                obv = indicators["obv"]
                vol_data.append([
                    "OBV",
                    obv.get("trend", "N/A"),
                    obv.get("signal", "N/A")
                ])
            
            if len(vol_data) > 1:
                vol_table = Table(vol_data, colWidths=[2*inch, 2*inch, 2*inch])
                vol_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLORS["warning"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light"]]),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                elements.append(vol_table)
        
        analysis_text = result.get("technical_analysis", "")
        if analysis_text:
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Analysis Summary", self.styles["heading2"]))
            elements.extend(self._parse_markdown(analysis_text))
        
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_fundamental_section(self, result: dict[str, Any]) -> list:
        """Create fundamental analysis section."""
        elements = []
        
        elements.append(Paragraph("Fundamental Analysis", self.styles["heading1"]))
        elements.append(HRFlowable(
            width="100%", thickness=1, color=COLORS["primary"],
            spaceBefore=0, spaceAfter=15
        ))
        
        ratios = result.get("financial_ratios", {})
        health = result.get("health_assessment", {})
        
        if health:
            health_score = health.get("score", 0)
            health_status = health.get("health", "N/A")
            
            elements.append(Paragraph(
                f"Financial Health: {health_status} ({health_score}/100)",
                self.styles["heading2"]
            ))
            
            factors = health.get("factors", [])
            if factors:
                for factor in factors:
                    elements.append(Paragraph(f"• {factor}", self.styles["body"]))
        
        if ratios:
            elements.append(Spacer(1, 10))
            elements.append(Paragraph("Key Ratios", self.styles["heading2"]))
            
            ratio_data = [["Metric", "Value", "Interpretation"]]
            
            for key, label in [
                ("pe_ratio", "P/E Ratio"),
                ("forward_pe", "Forward P/E"),
                ("dividend_yield", "Dividend Yield"),
                ("beta", "Beta"),
            ]:
                if key in ratios:
                    r = ratios[key]
                    value = r.get("value", "N/A")
                    if isinstance(value, float):
                        value = f"{value:.2f}"
                    interp = r.get("interpretation", "N/A")
                    ratio_data.append([label, str(value), interp[:40]])
            
            if len(ratio_data) > 1:
                ratio_table = Table(ratio_data, colWidths=[1.5*inch, 1.5*inch, 3*inch])
                ratio_table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLORS["success"]),
                    ("TEXTCOLOR", (0, 0), (-1, 0), COLORS["white"]),
                    ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, COLORS["light"]),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [COLORS["white"], COLORS["light"]]),
                ]))
                elements.append(ratio_table)
        
        analysis_text = result.get("fundamental_analysis", "")
        if analysis_text:
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Analysis Summary", self.styles["heading2"]))
            elements.extend(self._parse_markdown(analysis_text))
        
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_sentiment_section(self, result: dict[str, Any]) -> list:
        """Create sentiment analysis section."""
        elements = []
        
        elements.append(Paragraph("Sentiment Analysis", self.styles["heading1"]))
        elements.append(HRFlowable(
            width="100%", thickness=1, color=COLORS["primary"],
            spaceBefore=0, spaceAfter=15
        ))
        
        sentiment_score = result.get("sentiment_score", 0)
        sentiment_signal = result.get("sentiment_signal", {})
        
        score_color = COLORS["success"] if sentiment_score > 0.2 else (
            COLORS["danger"] if sentiment_score < -0.2 else COLORS["warning"]
        )
        
        elements.append(Paragraph(
            f"Sentiment Score: {sentiment_score:.2f}",
            ParagraphStyle("SentScore", fontSize=16, textColor=score_color,
                          fontName="Helvetica-Bold", spaceAfter=12)
        ))
        elements.append(Spacer(1, 8))
        elements.append(Paragraph(
            f"Signal: {sentiment_signal.get('signal', 'N/A')} "
            f"(Confidence: {sentiment_signal.get('confidence', 0):.0%})",
            self.styles["body"]
        ))
        
        analysis_text = result.get("sentiment_analysis", "")
        if analysis_text:
            elements.append(Spacer(1, 15))
            elements.append(Paragraph("Analysis Summary", self.styles["heading2"]))
            elements.extend(self._parse_markdown(analysis_text))
        
        elements.append(Spacer(1, 20))
        
        return elements

    def _create_thesis_section(self, result: dict[str, Any]) -> list:
        """Create investment thesis section."""
        elements = []
        
        elements.append(Paragraph("Investment Thesis", self.styles["heading1"]))
        elements.append(HRFlowable(
            width="100%", thickness=1, color=COLORS["primary"],
            spaceBefore=0, spaceAfter=15
        ))
        
        thesis = result.get("final_thesis", "")
        if thesis:
            elements.extend(self._parse_markdown(thesis))
        else:
            elements.append(Paragraph(
                "No investment thesis available.",
                self.styles["body"]
            ))
        
        elements.append(Spacer(1, 30))
        
        return elements

    def _create_disclaimer(self) -> list:
        """Create disclaimer section."""
        elements = []
        
        elements.append(HRFlowable(
            width="100%", thickness=1, color=COLORS["muted"],
            spaceBefore=20, spaceAfter=10
        ))
        
        disclaimer_text = """
        <b>DISCLAIMER:</b> This report is generated by an AI-powered multi-agent system 
        for informational purposes only. It does not constitute financial advice, 
        investment recommendations, or an offer to buy or sell securities. 
        Past performance is not indicative of future results. Always conduct your own 
        research and consult with a qualified financial advisor before making investment decisions.
        The creators of this system are not responsible for any financial losses incurred 
        based on the information provided in this report.
        """
        
        elements.append(Paragraph(
            disclaimer_text,
            ParagraphStyle("Disclaimer", fontSize=8, textColor=COLORS["muted"],
                          alignment=TA_JUSTIFY, leading=10)
        ))
        
        return elements

    def _add_header_footer(self, canvas, doc):
        """Add header and footer to each page."""
        canvas.saveState()
        
        canvas.setStrokeColor(COLORS["primary"])
        canvas.setLineWidth(2)
        canvas.line(doc.leftMargin, doc.height + doc.topMargin - 10,
                   doc.width + doc.leftMargin, doc.height + doc.topMargin - 10)
        
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(COLORS["muted"])
        canvas.drawString(doc.leftMargin, doc.height + doc.topMargin - 5, 
                         PROJECT_NAME.upper())
        
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(doc.width + doc.leftMargin, 
                              doc.height + doc.topMargin - 5,
                              datetime.now().strftime("%Y-%m-%d"))
        
        canvas.setStrokeColor(COLORS["light"])
        canvas.setLineWidth(0.5)
        canvas.line(doc.leftMargin, 0.5*inch,
                   doc.width + doc.leftMargin, 0.5*inch)
        
        # Show generated date above page number on first page
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(COLORS["muted"])
        if doc.page == 1:
            date_str = getattr(self, '_cover_date', datetime.now().strftime("%B %d, %Y"))
            canvas.drawCentredString(doc.width/2 + doc.leftMargin, 0.5*inch,
                                    f"Generated: {date_str}")
        canvas.drawCentredString(doc.width/2 + doc.leftMargin, 0.35*inch,
                                f"Page {doc.page}")
        
        canvas.restoreState()
