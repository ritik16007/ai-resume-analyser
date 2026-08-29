"""
AI Resume Analyzer - Streamlit App (v2)
Uses pypdf for PDF text extraction, reportlab for PDF generation, and
direct REST calls to the Gemini Generative Language API (v1) - no
google-genai/google-generativeai SDK.

Features:
- ATS resume analysis (score, skills, strengths, improvements, verdict)
- Resume vs Job Description match % comparison
- AI-generated cover letter
- Downloadable PDF of improved-resume suggestions and cover letter
- Modern landing page with a pricing section

Run with:
    python -m streamlit run app.py
"""

import io
import json
import re

import requests
import streamlit as st
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-pro",
]

API_BASE = "https://generativelanguage.googleapis.com/v1/models"

ATS_HEADERS = [
    "ATS_SCORE", "STRONG_SKILLS", "MISSING_SKILLS",
    "STRENGTHS", "IMPROVEMENTS", "FINAL_VERDICT",
]

MATCH_HEADERS = [
    "MATCH_PERCENT", "MATCHING_KEYWORDS", "MISSING_KEYWORDS",
    "GAP_SUMMARY", "SUGGESTIONS", "VERDICT",
]

ATS_PROMPT_TEMPLATE = """You are a professional ATS (Applicant Tracking System) Expert with 15+ years of experience in resume screening and recruitment.

Analyze the following resume text carefully:

---RESUME START---
{resume_text}
---RESUME END---

Give your output in Hinglish (Hindi+English mix, jaise log normally baat karte hain) with EXACTLY these sections, using the EXACT headers below so the output can be parsed programmatically:

### ATS_SCORE
(Give ONLY a single integer number between 0 and 100 representing the ATS compatibility score, nothing else on this line)

### STRONG_SKILLS
(Bullet list of skills already present and strong in the resume)

### MISSING_SKILLS
(Bullet list of important skills/keywords that are missing but expected for this profile)

### STRENGTHS
(Exactly 3 bullet points about the resume's strengths)

### IMPROVEMENTS
(Exactly 3 bullet points about what to improve, each with a concrete example of how to rewrite/fix it)

### FINAL_VERDICT
(2-3 lines overall verdict - should this resume pass ATS screening or not, and why)

Remember: Hinglish mein likho (mix Hindi + English naturally), professional tone rakho, aur ATS_SCORE section mein sirf number likho.
"""

JD_MATCH_PROMPT_TEMPLATE = """You are a professional ATS Expert and technical recruiter.

Compare the following RESUME against the JOB DESCRIPTION and evaluate how well they match.

---RESUME START---
{resume_text}
---RESUME END---

---JOB DESCRIPTION START---
{jd_text}
---JOB DESCRIPTION END---

Give your output in Hinglish (Hindi+English mix) with EXACTLY these sections, using the EXACT headers below so the output can be parsed programmatically:

### MATCH_PERCENT
(Give ONLY a single integer number between 0 and 100 representing how well the resume matches the job description, nothing else on this line)

### MATCHING_KEYWORDS
(Bullet list of skills/keywords from the JD that ARE present in the resume)

### MISSING_KEYWORDS
(Bullet list of important skills/keywords from the JD that are MISSING from the resume)

### GAP_SUMMARY
(2-3 lines summarizing the overall gap between the resume and this job's requirements)

### SUGGESTIONS
(Exactly 3 concrete, actionable bullet points on how to tailor the resume specifically for this job description)

### VERDICT
(2-3 lines: should this candidate apply, and how strong is their fit for this specific role)

Remember: Hinglish mein likho, professional tone rakho, aur MATCH_PERCENT section mein sirf number likho.
"""

COVER_LETTER_PROMPT_TEMPLATE = """You are a professional career coach and expert cover letter writer.

Using the resume below{jd_clause}, write a compelling, professional cover letter.

---RESUME START---
{resume_text}
---RESUME END---
{jd_block}
Requirements for the cover letter:
- Professional tone, in clear English (not Hinglish for this one, since it's a formal document)
- 3-4 paragraphs: opening hook, relevant experience/skills matched to the role, why this candidate is a great fit, strong closing with a call to action
- Do NOT invent facts, companies, or numbers that are not in the resume - only use what's actually there
- Keep it concise: around 250-350 words
- Do not include placeholder brackets like [Company Name] unless the job description names the company - if unknown, phrase it generically (e.g. "your organization")
- Output ONLY the cover letter text, no headers, no explanations, no markdown formatting
"""

# ----------------------------------------------------------------------------
# HELPERS - PDF text extraction
# ----------------------------------------------------------------------------


def extract_pdf_text(uploaded_file) -> str:
    """Extract text from an uploaded PDF file object."""
    try:
        reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        st.error(f"❌ PDF extraction failed: {e}")
        return ""


# ----------------------------------------------------------------------------
# HELPERS - Gemini API calls
# ----------------------------------------------------------------------------


def call_gemini_api(api_key: str, model: str, prompt: str, timeout: int = 60):
    """
    Call the Gemini generateContent REST endpoint for a specific model.
    Returns (success: bool, text_or_error: str, status_code: int|None)
    """
    url = f"{API_BASE}/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2048},
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)
    except requests.exceptions.Timeout:
        return False, "Request timed out. Please try again.", None
    except requests.exceptions.ConnectionError:
        return False, "Connection error. Check your internet connection.", None
    except Exception as e:
        return False, f"Unexpected request error: {e}", None

    status_code = response.status_code

    if status_code == 200:
        try:
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                feedback = data.get("promptFeedback", {})
                return False, f"No candidates returned. Feedback: {feedback}", status_code
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)
            if not text.strip():
                return False, "Empty response text from model.", status_code
            return True, text, status_code
        except Exception as e:
            return False, f"Failed to parse API response: {e}", status_code

    try:
        err_json = response.json()
        err_msg = err_json.get("error", {}).get("message", response.text)
    except Exception:
        err_msg = response.text

    return False, f"[{status_code}] {err_msg}", status_code


def run_gemini_with_fallback(api_key: str, prompt: str):
    """
    Try each model in MODEL_FALLBACK_CHAIN until one succeeds.
    Returns (success, result_text_or_error, model_used, attempts_log)
    """
    attempts_log = []

    for model in MODEL_FALLBACK_CHAIN:
        success, result, status_code = call_gemini_api(api_key, model, prompt)
        attempts_log.append(
            {"model": model, "success": success, "status_code": status_code,
             "message": None if success else result}
        )

        if success:
            return True, result, model, attempts_log

        if status_code in (401, 403):
            return False, f"Authentication error - check your API key.\n\nDetails: {result}", model, attempts_log
        if status_code == 429:
            return False, f"Rate limit / quota exceeded.\n\nDetails: {result}", model, attempts_log
        if status_code not in (404, 400, None):
            continue

    return False, "All models in the fallback chain failed. See attempt log below.", None, attempts_log


# ----------------------------------------------------------------------------
# HELPERS - Parsing structured AI output
# ----------------------------------------------------------------------------


def parse_sections(raw_text: str, headers: list) -> dict:
    """Generic parser: split raw_text into a dict keyed by ### HEADER markers."""
    sections = {h: "" for h in headers}
    header_alt = "|".join(headers)
    pattern = rf"###\s*({header_alt})\s*(.*?)(?=###\s*(?:{header_alt})|\Z)"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)

    for header, content in matches:
        key = header.strip().upper()
        if key in sections:
            sections[key] = content.strip()

    sections["_raw"] = raw_text
    sections["_parsed_ok"] = any(sections[k] for k in sections if not k.startswith("_"))
    return sections


def extract_percent(text: str):
    match = re.search(r"\d{1,3}", text)
    if match:
        return min(int(match.group()), 100)
    return None


def bullets_to_list(text: str) -> list:
    """Turn a bullet-style block of text into a clean list of strings."""
    items = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*•").strip()
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
        if cleaned:
            items.append(cleaned)
    return items or ([text.strip()] if text.strip() else [])


# ----------------------------------------------------------------------------
# HELPERS - PDF generation (downloadable output)
# ----------------------------------------------------------------------------


def _pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#4F46E5"),
            spaceBefore=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="DocTitle",
            parent=styles["Title"],
            textColor=colors.HexColor("#1F2937"),
        )
    )
    return styles


def generate_suggestions_pdf(parsed: dict) -> bytes:
    """Build a PDF of improved-resume suggestions from the parsed ATS analysis."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
    )
    styles = _pdf_styles()
    story = []

    story.append(Paragraph("Resume Improvement Report", styles["DocTitle"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Generated by AI Resume Analyzer", styles["Normal"]))
    story.append(Spacer(1, 16))

    score = extract_percent(parsed.get("ATS_SCORE", ""))
    if score is not None:
        story.append(Paragraph(f"ATS Score: {score} / 100", styles["SectionHeading"]))
        story.append(Spacer(1, 6))

    section_titles = {
        "STRONG_SKILLS": "Strong Skills",
        "MISSING_SKILLS": "Missing Skills to Add",
        "STRENGTHS": "Key Strengths",
        "IMPROVEMENTS": "Suggested Improvements",
        "FINAL_VERDICT": "Final Verdict",
    }

    for key, title in section_titles.items():
        content = parsed.get(key, "").strip()
        if not content:
            continue
        story.append(Paragraph(title, styles["SectionHeading"]))
        if key == "FINAL_VERDICT":
            story.append(Paragraph(content.replace("\n", "<br/>"), styles["Normal"]))
        else:
            items = bullets_to_list(content)
            list_items = [ListItem(Paragraph(item, styles["Normal"]), bulletColor=colors.HexColor("#4F46E5")) for item in items]
            story.append(ListFlowable(list_items, bulletType="bullet", start="•"))
        story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()


def generate_text_pdf(title: str, body_text: str) -> bytes:
    """Build a simple single-section PDF from a title and a block of plain text (used for cover letters)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=LETTER,
        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
        leftMargin=1 * inch, rightMargin=1 * inch,
    )
    styles = _pdf_styles()
    story = [Paragraph(title, styles["DocTitle"]), Spacer(1, 14)]

    for para in body_text.split("\n\n"):
        cleaned = para.strip().replace("\n", "<br/>")
        if cleaned:
            story.append(Paragraph(cleaned, styles["Normal"]))
            story.append(Spacer(1, 10))

    doc.build(story)
    return buffer.getvalue()


# ----------------------------------------------------------------------------
# SESSION STATE INIT
# ----------------------------------------------------------------------------

defaults = {
    "entered_app": False,
    "resume_text": "",
    "ats_result_raw": None,
    "ats_model_used": None,
    "jd_result_raw": None,
    "jd_model_used": None,
    "cover_letter_text": None,
    "cover_letter_model_used": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


def go_to_app():
    st.session_state.entered_app = True


def go_to_landing():
    st.session_state.entered_app = False


# ----------------------------------------------------------------------------
# LANDING PAGE
# ----------------------------------------------------------------------------


def render_landing_page():
    st.markdown(
        """
        <style>
        .hero-badge {
            display: inline-block; padding: 6px 16px; border-radius: 999px;
            background: linear-gradient(90deg, #6366F1, #8B5CF6); color: white;
            font-size: 0.85rem; font-weight: 600; margin-bottom: 16px;
        }
        .hero-title { font-size: 2.6rem; font-weight: 800; line-height: 1.15; margin-bottom: 8px; }
        .hero-subtitle { font-size: 1.15rem; opacity: 0.75; max-width: 700px; margin-bottom: 24px; }
        .feature-card {
            border-radius: 14px; padding: 22px; background: rgba(127,127,127,0.08);
            border: 1px solid rgba(127,127,127,0.15); height: 100%;
        }
        .feature-icon { font-size: 1.8rem; margin-bottom: 8px; }
        .feature-title { font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; }
        .feature-desc { font-size: 0.92rem; opacity: 0.75; }
        .price-card {
            border-radius: 16px; padding: 26px; background: rgba(127,127,127,0.06);
            border: 1px solid rgba(127,127,127,0.15); text-align: center; height: 100%;
        }
        .price-card.popular { border: 2px solid #6366F1; background: rgba(99,102,241,0.08); }
        .price-tier { font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7; }
        .price-amount { font-size: 2.2rem; font-weight: 800; margin: 10px 0; }
        .price-amount span { font-size: 1rem; font-weight: 400; opacity: 0.6; }
        .popular-tag {
            background: #6366F1; color: white; padding: 3px 12px; border-radius: 999px;
            font-size: 0.75rem; font-weight: 700; display: inline-block; margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Hero ---
    st.markdown('<div class="hero-badge">✨ Powered by Gemini AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title">Land your next job faster<br>with an ATS-ready resume</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Upload your resume, get an instant ATS score, match it against any '
        'job description, and generate a tailored cover letter — all in one place.</div>',
        unsafe_allow_html=True,
    )

    hero_col1, hero_col2, _ = st.columns([1, 1, 2])
    with hero_col1:
        st.button("🚀 Get Started Free", type="primary", use_container_width=True, on_click=go_to_app)
    with hero_col2:
        st.button("📖 See How It Works", use_container_width=True, on_click=go_to_app)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Features ---
    st.markdown("## Everything you need to get past the ATS")
    st.write("")

    features = [
        ("🎯", "ATS Score & Analysis", "Get an instant compatibility score with strong/missing skills, strengths, and concrete improvements."),
        ("📊", "Job Description Match", "Paste any JD and see your exact match %, plus which keywords you're missing."),
        ("✉️", "AI Cover Letter", "Generate a tailored, professional cover letter from your resume in seconds."),
        ("📥", "Downloadable Reports", "Export your improvement suggestions and cover letter as clean PDF files."),
    ]
    f_cols = st.columns(4)
    for col, (icon, title, desc) in zip(f_cols, features):
        with col:
            st.markdown(
                f"""<div class="feature-card">
                        <div class="feature-icon">{icon}</div>
                        <div class="feature-title">{title}</div>
                        <div class="feature-desc">{desc}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # --- Pricing ---
    st.markdown("## Simple, transparent pricing")
    st.write("Pick the plan that fits how often you're applying.")
    st.write("")

    price_cols = st.columns(3)

    with price_cols[0]:
        st.markdown(
            """<div class="price-card">
                    <div class="price-tier">Free</div>
                    <div class="price-amount">$0<span>/mo</span></div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("- 3 resume analyses / month\n- Basic ATS score\n- 1 JD match check\n- Community support")
        st.button("Start Free", key="price_free", use_container_width=True, on_click=go_to_app)

    with price_cols[1]:
        st.markdown(
            """<div class="price-card popular">
                    <div class="popular-tag">MOST POPULAR</div><br>
                    <div class="price-tier">Pro</div>
                    <div class="price-amount">$9<span>/mo</span></div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("- Unlimited resume analyses\n- Unlimited JD match checks\n- AI cover letter generator\n- PDF export of all reports\n- Priority support")
        st.button("Upgrade to Pro", key="price_pro", type="primary", use_container_width=True, on_click=go_to_app)

    with price_cols[2]:
        st.markdown(
            """<div class="price-card">
                    <div class="price-tier">Team</div>
                    <div class="price-amount">$29<span>/mo</span></div>
                </div>""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("- Everything in Pro\n- Up to 10 team seats\n- Shared templates\n- Usage analytics dashboard\n- Dedicated support")
        st.button("Contact Sales", key="price_team", use_container_width=True, on_click=go_to_app)

    st.caption(
        "Note: this demo app currently runs entirely on your own Gemini API key — "
        "pricing above is illustrative UI only and no payment is processed."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit + Gemini API | Not affiliated with Google")
    st.caption("made by Ritik Kumar Parida ❤️")

# ----------------------------------------------------------------------------
# MAIN APP PAGE
# ----------------------------------------------------------------------------


def render_sidebar():
    with st.sidebar:
        st.button("🏠 Back to Home", use_container_width=True, on_click=go_to_landing)
        st.markdown("---")
        st.title("⚙️ Settings")
        st.markdown("### Gemini API Key")
        api_key = st.text_input(
            "Enter your Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Your key is only used for this session and never stored.",
        )
        st.markdown("🔑 [Get your free API key here](https://aistudio.google.com/app/apikey)")

        st.markdown("---")
        st.markdown("### ℹ️ How it works")
        st.markdown(
            """
            1. Enter your Gemini API key above
            2. Upload your resume (PDF)
            3. Run ATS Analysis, JD Match, or generate a Cover Letter
            4. Download your results as PDF
            """
        )

        st.markdown("---")
        st.markdown("### 🤖 Model fallback order")
        st.code("\n".join(MODEL_FALLBACK_CHAIN), language=None)
        st.caption("App automatically tries the next model if one isn't available.")

    return api_key


def render_upload_section():
    st.subheader("1️⃣ Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload your resume (PDF only)", type=["pdf"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_pdf_text(uploaded_file)
            st.session_state.resume_text = extracted_text

        if extracted_text:
            st.success(f"✅ Extracted {len(extracted_text)} characters from PDF")
            with st.expander("📃 View Extracted Resume Text"):
                st.text_area("Extracted Text", extracted_text, height=220, label_visibility="collapsed")
        else:
            st.error(
                "⚠️ Could not extract any text from this PDF. "
                "It might be a scanned/image-based PDF. Try a text-based PDF instead."
            )


def render_ats_tab(api_key: str):
    st.subheader("🎯 ATS Resume Analysis")

    disabled = not (api_key and st.session_state.resume_text)
    if not api_key:
        st.warning("👈 Please enter your Gemini API key in the sidebar first.")
    elif not st.session_state.resume_text:
        st.info("👆 Please upload a PDF resume above to continue.")

    if st.button("🚀 Analyze with Gemini AI", type="primary", disabled=disabled, key="ats_btn"):
        with st.spinner("🔍 Analyzing your resume with Gemini AI..."):
            prompt = ATS_PROMPT_TEMPLATE.format(resume_text=st.session_state.resume_text[:15000])
            success, result, model_used, attempts_log = run_gemini_with_fallback(api_key, prompt)

        if success:
            st.session_state.ats_result_raw = result
            st.session_state.ats_model_used = model_used
            st.success(f"✅ Analysis complete! (Model used: `{model_used}`)")
        else:
            st.session_state.ats_result_raw = None
            st.error(f"❌ Analysis failed: {result}")
            with st.expander("🔧 See detailed attempt log (debug info)"):
                for attempt in attempts_log:
                    icon = "✅" if attempt["success"] else "❌"
                    st.markdown(f"{icon} **{attempt['model']}** — status: `{attempt['status_code']}`")
                    if attempt["message"]:
                        st.caption(attempt["message"])

    if not st.session_state.ats_result_raw:
        return

    parsed = parse_sections(st.session_state.ats_result_raw, ATS_HEADERS)
    st.markdown("---")
    st.caption(f"Generated using model: `{st.session_state.ats_model_used}`")

    if not parsed["_parsed_ok"]:
        st.warning("⚠️ Couldn't parse structured sections. Showing raw output instead.")
        st.markdown(parsed["_raw"])
        return

    score = extract_percent(parsed["ATS_SCORE"])
    score_col1, score_col2 = st.columns([1, 2])
    with score_col1:
        st.metric("ATS Score", f"{score}/100" if score is not None else "N/A")
    with score_col2:
        if score is not None:
            st.progress(score / 100)
            if score >= 75:
                st.success("🟢 Excellent! Resume is well optimized for ATS.")
            elif score >= 50:
                st.warning("🟡 Decent, but there's room for improvement.")
            else:
                st.error("🔴 Needs significant improvement to pass ATS screening.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Strong Skills")
        st.markdown(parsed["STRONG_SKILLS"] or "_Not provided_")
    with c2:
        st.subheader("❌ Missing Skills")
        st.markdown(parsed["MISSING_SKILLS"] or "_Not provided_")

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("💪 Top Strengths")
        st.markdown(parsed["STRENGTHS"] or "_Not provided_")
    with c4:
        st.subheader("🔧 Top Improvements")
        st.markdown(parsed["IMPROVEMENTS"] or "_Not provided_")

    st.markdown("---")
    st.subheader("🏁 Final Verdict")
    st.info(parsed["FINAL_VERDICT"] or "_Not provided_")

    st.markdown("---")
    pdf_bytes = generate_suggestions_pdf(parsed)
    st.download_button(
        "📥 Download Improved Resume Suggestions (PDF)",
        data=pdf_bytes,
        file_name="resume_improvement_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("📄 View Raw AI Response"):
        st.text(parsed["_raw"])


def render_jd_match_tab(api_key: str):
    st.subheader("📊 Resume vs Job Description Match")

    jd_text = st.text_area(
        "Paste the Job Description here",
        height=200,
        placeholder="Paste the full job description text you're applying for...",
        key="jd_text_input",
    )

    disabled = not (api_key and st.session_state.resume_text and jd_text.strip())
    if not api_key:
        st.warning("👈 Please enter your Gemini API key in the sidebar first.")
    elif not st.session_state.resume_text:
        st.info("👆 Please upload a PDF resume above to continue.")
    elif not jd_text.strip():
        st.info("📋 Paste a job description above to compare it against your resume.")

    if st.button("🔍 Compare Resume with JD", type="primary", disabled=disabled, key="jd_btn"):
        with st.spinner("🔍 Comparing resume with job description..."):
            prompt = JD_MATCH_PROMPT_TEMPLATE.format(
                resume_text=st.session_state.resume_text[:15000],
                jd_text=jd_text[:8000],
            )
            success, result, model_used, attempts_log = run_gemini_with_fallback(api_key, prompt)

        if success:
            st.session_state.jd_result_raw = result
            st.session_state.jd_model_used = model_used
            st.success(f"✅ Comparison complete! (Model used: `{model_used}`)")
        else:
            st.session_state.jd_result_raw = None
            st.error(f"❌ Comparison failed: {result}")
            with st.expander("🔧 See detailed attempt log (debug info)"):
                for attempt in attempts_log:
                    icon = "✅" if attempt["success"] else "❌"
                    st.markdown(f"{icon} **{attempt['model']}** — status: `{attempt['status_code']}`")
                    if attempt["message"]:
                        st.caption(attempt["message"])

    if not st.session_state.jd_result_raw:
        return

    parsed = parse_sections(st.session_state.jd_result_raw, MATCH_HEADERS)
    st.markdown("---")
    st.caption(f"Generated using model: `{st.session_state.jd_model_used}`")

    if not parsed["_parsed_ok"]:
        st.warning("⚠️ Couldn't parse structured sections. Showing raw output instead.")
        st.markdown(parsed["_raw"])
        return

    match_pct = extract_percent(parsed["MATCH_PERCENT"])
    m1, m2 = st.columns([1, 2])
    with m1:
        st.metric("Match %", f"{match_pct}%" if match_pct is not None else "N/A")
    with m2:
        if match_pct is not None:
            st.progress(match_pct / 100)
            if match_pct >= 75:
                st.success("🟢 Strong match — you're a great fit for this role.")
            elif match_pct >= 50:
                st.warning("🟡 Partial match — tailor your resume before applying.")
            else:
                st.error("🔴 Low match — significant gaps for this specific role.")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("✅ Matching Keywords")
        st.markdown(parsed["MATCHING_KEYWORDS"] or "_Not provided_")
    with c2:
        st.subheader("❌ Missing Keywords")
        st.markdown(parsed["MISSING_KEYWORDS"] or "_Not provided_")

    st.markdown("---")
    st.subheader("📌 Gap Summary")
    st.markdown(parsed["GAP_SUMMARY"] or "_Not provided_")

    st.markdown("---")
    st.subheader("🛠️ Suggestions to Tailor Your Resume")
    st.markdown(parsed["SUGGESTIONS"] or "_Not provided_")

    st.markdown("---")
    st.subheader("🏁 Verdict")
    st.info(parsed["VERDICT"] or "_Not provided_")

    with st.expander("📄 View Raw AI Response"):
        st.text(parsed["_raw"])


def render_cover_letter_tab(api_key: str):
    st.subheader("✉️ AI Cover Letter Generator")

    use_jd = st.checkbox("Tailor this cover letter to a specific job description (optional)")
    jd_text = ""
    if use_jd:
        jd_text = st.text_area(
            "Paste the Job Description here",
            height=180,
            placeholder="Paste the job description to tailor the letter...",
            key="cover_jd_input",
        )

    disabled = not (api_key and st.session_state.resume_text)
    if not api_key:
        st.warning("👈 Please enter your Gemini API key in the sidebar first.")
    elif not st.session_state.resume_text:
        st.info("👆 Please upload a PDF resume above to continue.")

    if st.button("✍️ Generate Cover Letter", type="primary", disabled=disabled, key="cover_btn"):
        with st.spinner("✍️ Writing your cover letter..."):
            if use_jd and jd_text.strip():
                jd_clause = " and the job description below"
                jd_block = f"\n---JOB DESCRIPTION START---\n{jd_text[:8000]}\n---JOB DESCRIPTION END---\n"
            else:
                jd_clause = ""
                jd_block = "\n"

            prompt = COVER_LETTER_PROMPT_TEMPLATE.format(
                resume_text=st.session_state.resume_text[:15000],
                jd_clause=jd_clause,
                jd_block=jd_block,
            )
            success, result, model_used, attempts_log = run_gemini_with_fallback(api_key, prompt)

        if success:
            st.session_state.cover_letter_text = result.strip()
            st.session_state.cover_letter_model_used = model_used
            st.success(f"✅ Cover letter generated! (Model used: `{model_used}`)")
        else:
            st.session_state.cover_letter_text = None
            st.error(f"❌ Generation failed: {result}")
            with st.expander("🔧 See detailed attempt log (debug info)"):
                for attempt in attempts_log:
                    icon = "✅" if attempt["success"] else "❌"
                    st.markdown(f"{icon} **{attempt['model']}** — status: `{attempt['status_code']}`")
                    if attempt["message"]:
                        st.caption(attempt["message"])

    if not st.session_state.cover_letter_text:
        return

    st.markdown("---")
    st.caption(f"Generated using model: `{st.session_state.cover_letter_model_used}`")
    st.text_area("Your Cover Letter", st.session_state.cover_letter_text, height=350)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "📥 Download as TXT",
            data=st.session_state.cover_letter_text,
            file_name="cover_letter.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with dl_col2:
        pdf_bytes = generate_text_pdf("Cover Letter", st.session_state.cover_letter_text)
        st.download_button(
            "📥 Download as PDF",
            data=pdf_bytes,
            file_name="cover_letter.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def render_app_page():
    api_key = render_sidebar()

    st.title("📄 AI Resume Analyzer")
    st.markdown(
        "##### Get instant ATS score, JD match %, aur AI cover letter — powered by Gemini AI"
    )
    st.markdown("---")

    render_upload_section()
    st.markdown("---")

    st.subheader("2️⃣ Choose an Action")
    tab1, tab2, tab3 = st.tabs(["🎯 ATS Analysis", "📊 JD Match", "✉️ Cover Letter"])
    with tab1:
        render_ats_tab(api_key)
    with tab2:
        render_jd_match_tab(api_key)
    with tab3:
        render_cover_letter_tab(api_key)

    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit + Gemini API | Not affiliated with Google")


# ----------------------------------------------------------------------------
# ROUTER
# ----------------------------------------------------------------------------

if st.session_state.entered_app:
    render_app_page()
else:
    render_landing_page()
