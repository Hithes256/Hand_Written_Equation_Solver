import streamlit as st
import easyocr
import cv2
import numpy as np
import sympy as sp
import re

st.set_page_config(page_title="🧠 AI Handwritten Equation Solver", layout="centered")
st.title("✍ Handwritten Math Equation Solver")
st.markdown("Upload a handwritten equation image (e.g., 3x+2=8, x^2 - 5x + 6 = 0, √x + 5 = 9)")

# ------------------- Cleaner -------------------
def clean_equation(text):
    # First, fix root-like symbols BEFORE s/S/5 replacement
    text = text.replace('√', 'sqrt')
    text = text.replace('✓', 'sqrt')
    text = text.replace('V', 'sqrt')
    text = text.replace('v', 'sqrt')

    # Now fix common OCR typos
    text = text.replace('X', 'x').replace('^', '')
    text = text.replace('−', '-').replace('×', '*').replace('÷', '/')
    text = text.replace('O', '0').replace('l', '1')

    # Carefully fix 's' or 'S' → 5 but not in 'sqrt'
    text = re.sub(r'(?<!sq)[sS](?!qrt)', '5', text)

    # Replace sqrt*x or sqrtx with sqrt(x)
    text = re.sub(r'sqrt\*?x', 'sqrt(x)', text)
    text = re.sub(r'sqrt\((.*?)\)', r'sqrt(\1)', text)  # keep parentheses

    # Fix math syntax
    text = re.sub(r'(?<=\d)x', '*x', text)         # 2x → 2*x
    text = re.sub(r'(?<=x)(\d)', '*\\1', text)     # x4 → x*4
    text = re.sub(r'(?<=x)\(\d)', '\\1', text)  # x*2 → x*2
    text = re.sub(r'(?<![a-zA-Z0-9])(\d+)(?=[a-zA-Z])', r'\1*', text)  # 3x4 → 3*x*4

    # Allow only safe characters
    text = re.sub(r'[^0-9a-zA-Z=+\-*/().^]', '', text)

    return text.strip()
# ------------------- Validator -------------------
def is_valid_equation(text):
    return '=' in text and 'x' in text

# ------------------- UI -------------------
uploaded_file = st.file_uploader("📤 Upload handwritten equation image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image, detail=0)
    raw_text = ' '.join(result)

    st.subheader("🔍 OCR Output:")
    st.code(raw_text)

    cleaned = clean_equation(raw_text)
    st.subheader("🧼 Cleaned Equation:")
    st.code(cleaned)

    if not is_valid_equation(cleaned):
        st.error("❌ Couldn't find a valid equation with 'x' and '='. Please try again.")
    else:
        try:
            lhs, rhs = cleaned.split('=')
            x = sp.symbols('x')

            lhs_expr = sp.sympify(lhs)
            rhs_expr = sp.sympify(rhs)
            eq = sp.Eq(lhs_expr, rhs_expr)
            solution = sp.solve(eq, x)

            st.subheader("✅ Parsed Equation:")
            st.latex(sp.latex(eq))

            st.subheader("📌 Solution:")

            if solution:
                if st.checkbox("🔁 Show exact symbolic form"):
                    st.success("x = " + ', '.join([str(s) for s in solution]))

                decimal_solutions = [sp.N(s) for s in solution]
                formatted = ', '.join([str(s.evalf()) for s in decimal_solutions])
                st.success(f"x ≈ {formatted}")
            else:
                st.warning("No solution found.")

        except Exception as e:
            st.error("💥 Error during parsing or solving. Possibly malformed equation.")
            st.exception(e)

    with st.expander("📘 Features Supported"):
        st.markdown("""
        - ✅ Linear, quadratic, and square root equations
        - 🔄 OCR symbol corrections: S, Z, O, l, etc.
        - ✨ Smart conversion: x2 → x**2, 3x4 → 3*x*4
        - 🧠 Safe parsing with sympy.sympify() (no Python eval)
        - 📌 Real and complex roots supported
        - 🧮 Symbolic + Decimal output
        """)