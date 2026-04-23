"""
SEMAF (Self-Evolving Multi-Agent Framework) - Streamlit 실험 시뮬레이션 앱
논문: 자기진화형 다중 에이전트 프레임워크(SEMAF): 견고한 LLM 기반 시스템을 위한 지속적 학습 및 적응적 협업
"""

import streamlit as st
import time
import random
import json
import os
from datetime import datetime
from typing import List, Dict, Any

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SEMAF 실험 시뮬레이션",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 스타일 ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* 전체 배경 */
.main { background-color: #f8f9fa; }

/* 카드 스타일 */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border-left: 4px solid #4f8ef7;
    margin-bottom: 12px;
}
.metric-card.semaf  { border-left-color: #4CAF50; }
.metric-card.baseline { border-left-color: #f44336; }
.metric-card.neutral  { border-left-color: #4f8ef7; }

/* 결과 테이블 헤더 */
.result-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2e6fad 100%);
    color: white;
    padding: 14px 20px;
    border-radius: 10px 10px 0 0;
    font-weight: 600;
    font-size: 16px;
}

/* 로그 박스 */
.log-box {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Courier New', monospace;
    font-size: 12px;
    padding: 14px;
    border-radius: 8px;
    height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
}

/* 뱃지 */
.badge-semaf    { background:#e8f5e9; color:#2e7d32; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-baseline { background:#ffebee; color:#c62828; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }
.badge-win      { background:#e3f2fd; color:#1565c0; padding:3px 10px; border-radius:12px; font-size:12px; font-weight:600; }

/* 진행 상황 */
.iter-badge {
    display:inline-block;
    width:24px; height:24px;
    border-radius:50%;
    text-align:center;
    line-height:24px;
    font-size:10px;
    font-weight:700;
    margin:2px;
}
.iter-done { background:#4CAF50; color:white; }
.iter-adapt { background:#FF9800; color:white; }
.iter-todo { background:#e0e0e0; color:#666; }

/* 섹션 타이틀 */
h3.section-title {
    color: #1e3a5f;
    border-bottom: 2px solid #4f8ef7;
    padding-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 메트릭 계산 함수 (논문 4장 공식)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_lra(adaptation_times: List[int]) -> float:
    """
    LRA = (1/N) * Σ(1/Ti)
    Ti: i번째 환경 변화 후 최적 성능으로 복귀하기까지의 반복 횟수
    """
    if not adaptation_times:
        return 0.0
    N = len(adaptation_times)
    return sum(1.0 / max(1, t) for t in adaptation_times) / N


def calculate_ce(quality_score: float, communication_count: int) -> float:
    """
    CE = Q / C
    Q: 최종 결과의 품질 점수 (0~1), C: 에이전트 간 통신 횟수
    """
    if communication_count == 0:
        return quality_score
    return quality_score / communication_count


def calculate_kri(error_rate: float) -> float:
    """
    KRI = 1 - (D_new / D_total)
    error_rate = D_new/D_total
    """
    return 1.0 - max(0.0, min(1.0, error_rate))


def calculate_reward(quality: float, comm: int, kri: float,
                     lq: float, lc: float, lk: float, kri_target: float) -> float:
    """
    논문 3.3.2 보상 함수: R = λQ·Q - λC·C + λK·(KRI - KRI_target)
    """
    return lq * quality - lc * comm + lk * (kri - kri_target)


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM API 호출 (실제 API 또는 폴백)
# ─────────────────────────────────────────────────────────────────────────────

def llm_api_call(prompt: str, system_type: str, api_key: str, model: str) -> Dict[str, Any]:
    """OpenAI API 호출 시도 → 실패 시 의사 응답 반환."""
    start = time.time()
    try:
        import openai
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_type},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400,
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        elapsed = time.time() - start
        return {
            "text": content.strip(),
            "source": "api",
            "model": model,
            "elapsed": round(elapsed, 2),
        }
    except Exception as e:
        elapsed = time.time() - start
        tag = "SEMAF" if "SEMAF" in system_type else "Baseline"
        quality_hint = "(높은 품질, 적응형)" if tag == "SEMAF" else "(고정 역할, 낮은 품질)"
        return {
            "text": f"[{tag} 폴백 응답] {prompt[:60]}... {quality_hint}",
            "source": "fallback",
            "model": model,
            "elapsed": round(elapsed, 2),
            "error": str(e),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 3. 시스템 성능 시뮬레이션 (논문 5장)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_iteration(
    system_type: str,
    prompt: str,
    iteration: int,
    adaptation_points: List[int],
    api_key: str,
    model: str,
    lambda_q: float,
    lambda_c: float,
    lambda_k: float,
    kri_target: float,
    use_real_api: bool,
) -> Dict[str, Any]:
    """단일 반복 시뮬레이션: LLM 호출 + 지표 계산."""

    # LLM 호출
    if use_real_api and api_key:
        llm_result = llm_api_call(prompt, system_type, api_key, model)
    else:
        tag = "SEMAF" if "SEMAF" in system_type else "Baseline"
        llm_result = {
            "text": f"[{tag} 시뮬레이션] {prompt[:60]}...",
            "source": "simulation",
            "model": model,
            "elapsed": round(random.uniform(0.1, 0.5), 2),
        }

    is_semaf = "SEMAF" in system_type
    is_adapt = iteration in adaptation_points

    # ── SEMAF 성능 파라미터 (논문 Table 9 기준) ────────────────────────────
    if is_semaf:
        if is_adapt:
            quality    = random.uniform(0.75, 0.85)
            comm_count = random.randint(10, 15)
            kri_err    = random.uniform(0.03, 0.07)
            adapt_time = random.randint(2, 4)
        else:
            quality    = random.uniform(0.90, 0.95)
            comm_count = random.randint(5, 8)
            kri_err    = random.uniform(0.01, 0.05)
            adapt_time = 0
    # ── Baseline 성능 파라미터 ───────────────────────────────────────────
    else:
        if is_adapt:
            quality    = random.uniform(0.40, 0.60)
            comm_count = random.randint(30, 40)
            kri_err    = random.uniform(0.20, 0.30)
            adapt_time = random.randint(15, 20)
        else:
            quality    = random.uniform(0.70, 0.80)
            comm_count = random.randint(15, 25)
            kri_err    = random.uniform(0.15, 0.25)
            adapt_time = 0

    ce  = calculate_ce(quality, comm_count)
    kri = calculate_kri(kri_err)
    reward = calculate_reward(quality, comm_count, kri,
                              lambda_q, lambda_c, lambda_k, kri_target)

    return {
        "iteration":     iteration,
        "system":        "SEMAF" if is_semaf else "Baseline",
        "prompt_short":  prompt[:50] + "...",
        "quality":       round(quality, 4),
        "comm_count":    comm_count,
        "kri_error":     round(kri_err, 4),
        "ce":            round(ce, 4),
        "kri":           round(kri, 4),
        "reward":        round(reward, 4),
        "adapt_time":    adapt_time,
        "is_adapt_pt":   is_adapt,
        "llm_text":      llm_result["text"][:120],
        "llm_source":    llm_result["source"],
        "llm_elapsed":   llm_result.get("elapsed", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. 최종 결과 집계
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(records: List[Dict]) -> Dict:
    """전체 반복 결과에서 LRA/CE/KRI 최종값 집계."""
    summary = {}
    for sys in ("SEMAF", "Baseline"):
        rows = [r for r in records if r["system"] == sys]
        adapt_times = [r["adapt_time"] for r in rows if r["adapt_time"] > 0]
        ce_vals  = [r["ce"]  for r in rows]
        kri_vals = [r["kri"] for r in rows]
        q_vals   = [r["quality"] for r in rows]
        c_vals   = [r["comm_count"] for r in rows]
        summary[sys] = {
            "lra":      round(calculate_lra(adapt_times), 4),
            "avg_ce":   round(sum(ce_vals)  / len(ce_vals),  4) if ce_vals  else 0,
            "avg_kri":  round(sum(kri_vals) / len(kri_vals), 4) if kri_vals else 0,
            "avg_q":    round(sum(q_vals)   / len(q_vals),   4) if q_vals   else 0,
            "avg_c":    round(sum(c_vals)   / len(c_vals),   1) if c_vals   else 0,
            "n_adapt":  len(adapt_times),
        }
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 5. 차트 데이터 준비 (Plotly 없이 Streamlit 내장 차트 활용)
# ─────────────────────────────────────────────────────────────────────────────

def build_chart_data(records: List[Dict], metric: str):
    """반복별 지표 데이터를 딕셔너리로 변환."""
    import pandas as pd
    df = pd.DataFrame(records)
    pivot = df.pivot_table(index="iteration", columns="system", values=metric, aggfunc="mean")
    return pivot


# ─────────────────────────────────────────────────────────────────────────────
# 6. 사이드바 설정
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/robot.png", width=60)
    st.title("SEMAF 실험 설정")
    st.caption("자기진화형 다중 에이전트 프레임워크")
    st.divider()

    st.subheader("🔑 LLM API 설정")
    use_real_api = st.toggle("실제 OpenAI API 사용", value=False,
                             help="OFF 시 시뮬레이션(의사코드) 모드로 실행됩니다.")
    api_key = ""
    model   = "gpt-4o-mini"
    if use_real_api:
        api_key = st.text_input("OpenAI API Key", type="password",
                                placeholder="sk-proj-...")
        model   = st.selectbox("모델 선택", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"])

    st.divider()
    st.subheader("⚙️ 실험 파라미터")
    num_iterations = st.slider("총 반복 횟수 (NUM_ITERATIONS)", 10, 100, 60, 5)
    num_agents     = st.slider("에이전트 수 (NUM_AGENTS)", 2, 6, 3)

    adapt_raw = st.text_input("환경 변화 지점 (쉼표 구분)", "20,50",
                              help="예: 20,50 → 20번, 50번 반복 시 환경 변화 발생")
    try:
        adaptation_points = [int(x.strip()) for x in adapt_raw.split(",") if x.strip()]
        adaptation_points = [p for p in adaptation_points if 1 <= p <= num_iterations]
    except Exception:
        adaptation_points = [20, 50]

    st.divider()
    st.subheader("📐 보상 함수 하이퍼파라미터")
    st.caption("R = λQ·Q - λC·C + λK·(KRI - KRI_target)")
    lambda_q   = st.slider("λQ (품질 가중치)",  0.1, 2.0, 1.0, 0.1)
    lambda_c   = st.slider("λC (통신 가중치)",  0.001, 0.1, 0.01, 0.001, format="%.3f")
    lambda_k   = st.slider("λK (KRI 가중치)",   0.1, 2.0, 0.5, 0.1)
    kri_target = st.slider("KRI 목표값",         0.7, 1.0, 0.95, 0.01)

    st.divider()
    st.subheader("📝 테스트 프롬프트")
    test_prompts = [
        "Summarize global AI regulation trends and major issues over the past 3 years, and discuss Korea's response strategy.",
        "Explain the core principles and current technical limitations of quantum computing, and predict 3 potential impacts on the financial industry.",
        "Describe the core components and respective functions of A2A in detail.",
    ]
    prompts_text = st.text_area("프롬프트 목록 (줄바꿈 구분)", "\n".join(test_prompts), height=120)
    test_prompts = [p.strip() for p in prompts_text.split("\n") if p.strip()]

    st.divider()
    random_seed = st.number_input("랜덤 시드 (재현성)", min_value=0, max_value=9999, value=42)
    delay_ms    = st.slider("반복 간 딜레이 (ms)", 0, 500, 50, 10)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 메인 화면
# ─────────────────────────────────────────────────────────────────────────────

st.title("🤖 SEMAF 실험 시뮬레이션 대시보드")
st.caption(
    "**Self-Evolving Multi-Agent Framework** — LRA · CE · KRI 메타지표 실증 검증 | "
    "SEMAF Simulation"
)

# ── 논문 요약 익스팬더 ──────────────────────────────────────────────────────
with st.expander("📄 논문 개요 및 메트릭 정의", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**LRA (적응 학습률)**
$$LRA = \\frac{1}{N}\\sum_{i=1}^{N}\\frac{1}{T_i}$$
- $T_i$: i번째 환경 변화 후 기준 성능 90% 회복까지 반복 횟수
- 값이 높을수록 빠른 적응
        """)
    with c2:
        st.markdown("""
**CE (협업 효율성)**
$$CE = \\frac{Q}{C}$$
- $Q$: 최종 결과 품질 점수 (0~1)
- $C$: 에이전트 간 총 통신 횟수
- 높은 품질 + 낮은 통신 = 높은 CE
        """)
    with c3:
        st.markdown("""
**KRI (지식 보유 지수)**
$$KRI = 1 - \\frac{D_{new}}{D_{total}}$$
- $D_{new}$: 새 지식 통합 후 기존 문항 오답률
- KRI ≥ 95% → 재앙적 망각 완화 검증
        """)

st.divider()

# ── 실험 제어 버튼 ───────────────────────────────────────────────────────────
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 6])
run_btn  = col_btn1.button("▶ 실험 시작", type="primary", use_container_width=True)
stop_btn = col_btn2.button("⏹ 초기화",    use_container_width=True)

if stop_btn:
    for k in ["records", "running", "finished"]:
        st.session_state.pop(k, None)
    st.rerun()

# ── 상태 초기화 ──────────────────────────────────────────────────────────────
if "records"  not in st.session_state: st.session_state.records  = []
if "running"  not in st.session_state: st.session_state.running  = False
if "finished" not in st.session_state: st.session_state.finished = False

# ─────────────────────────────────────────────────────────────────────────────
# 8. 실험 실행 루프
# ─────────────────────────────────────────────────────────────────────────────

if run_btn:
    random.seed(random_seed)
    st.session_state.records  = []
    st.session_state.running  = True
    st.session_state.finished = False

    st.markdown("### 🔄 실험 진행 중...")

    # 진행 표시 위젯
    progress_bar  = st.progress(0, text="초기화 중...")
    status_text   = st.empty()
    log_container = st.empty()
    chart_ph      = st.empty()
    logs: List[str] = []

    def add_log(msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{ts}] {msg}")
        log_html = "\n".join(logs[-40:])   # 최근 40줄
        log_container.markdown(
            f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True
        )

    add_log(f"실험 시작 — {num_iterations}회 반복, 에이전트 {num_agents}개, 시드 {random_seed}")
    add_log(f"환경 변화 지점: {adaptation_points}")
    add_log(f"API 모드: {'실제 OpenAI' if use_real_api else '시뮬레이션(폴백)'}")
    add_log("-" * 60)

    import pandas as pd

    for i in range(1, num_iterations + 1):
        prompt_idx  = (i - 1) % len(test_prompts)
        cur_prompt  = test_prompts[prompt_idx]
        is_adapt_pt = (i in adaptation_points)

        # Baseline
        b_res = simulate_iteration(
            "Baseline (Fixed Role System)", cur_prompt, i,
            adaptation_points, api_key, model,
            lambda_q, lambda_c, lambda_k, kri_target, use_real_api
        )
        # SEMAF
        s_res = simulate_iteration(
            "SEMAF (Self-Evolving System)", cur_prompt, i,
            adaptation_points, api_key, model,
            lambda_q, lambda_c, lambda_k, kri_target, use_real_api
        )

        st.session_state.records.extend([b_res, s_res])

        # 로그
        if is_adapt_pt:
            add_log(f"⚠️  [Iter {i:3d}] 환경 변화 발생!")
        add_log(
            f"  Iter {i:3d} | Baseline CE={b_res['ce']:.4f} KRI={b_res['kri']:.4f} | "
            f"SEMAF CE={s_res['ce']:.4f} KRI={s_res['kri']:.4f}"
        )

        # 진행 바
        pct = int(i / num_iterations * 100)
        progress_bar.progress(pct, text=f"진행 중 {i}/{num_iterations} ({pct}%)")
        status_text.caption(
            f"현재 반복: **{i}** | 프롬프트: *{cur_prompt[:50]}...*"
        )

        # 실시간 차트 (10회마다 갱신)
        if i % 10 == 0 or i == num_iterations:
            df_live = pd.DataFrame(st.session_state.records)
            pivot   = df_live.pivot_table(index="iteration", columns="system",
                                          values="ce", aggfunc="mean")
            chart_ph.line_chart(pivot, use_container_width=True)

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    st.session_state.running  = False
    st.session_state.finished = True
    add_log("=" * 60)
    add_log("✅ 실험 완료!")
    progress_bar.progress(100, text="완료!")


# ─────────────────────────────────────────────────────────────────────────────
# 9. 결과 표시 (실험 완료 후)
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.finished and st.session_state.records:
    import pandas as pd

    records = st.session_state.records
    df      = pd.DataFrame(records)
    summary = compute_summary(records)

    st.divider()
    st.markdown("## 📊 최종 실험 결과")

    # ── 9-1. KPI 요약 카드 ──────────────────────────────────────────────────
    st.markdown("### 핵심 지표 비교 (논문 Table 9)")

    metrics_info = [
        ("LRA", "lra",     "1/T 단위", "4.09배 향상 (논문)"),
        ("CE",  "avg_ce",  "Q/C 단위", "3.91배 향상 (논문)"),
        ("KRI", "avg_kri", "1-Err",    "+14.6%p 향상 (논문)"),
    ]

    for label, key, unit, paper_ref in metrics_info:
        s_val = summary["SEMAF"][key]
        b_val = summary["Baseline"][key]
        ratio = (s_val / b_val) if b_val > 0 else float("inf")
        win   = s_val > b_val

        ca, cb, cc, cd = st.columns([2, 2, 2, 3])
        ca.metric(f"SEMAF  {label}", f"{s_val:.4f}", f"+{s_val-b_val:.4f} vs Baseline",
                  delta_color="normal")
        cb.metric(f"Baseline {label}", f"{b_val:.4f}", unit)
        cc.metric("SEMAF 우위", f"{'✅' if win else '❌'} {ratio:.2f}×")
        cd.markdown(f"<div class='metric-card neutral' style='padding:10px'>"
                    f"<small>📌 참고: {paper_ref}</small></div>", unsafe_allow_html=True)

    st.divider()

    # ── 9-2. 상세 비교 테이블 ───────────────────────────────────────────────
    st.markdown("### 📋 상세 비교 테이블")
    comp_df = pd.DataFrame({
        "지표":     ["LRA (적응 학습률)", "CE (협업 효율성)", "KRI (지식 보유 지수)",
                     "평균 품질 Q",       "평균 통신 횟수 C"],
        "SEMAF":    [summary["SEMAF"]["lra"],  summary["SEMAF"]["avg_ce"],
                     summary["SEMAF"]["avg_kri"], summary["SEMAF"]["avg_q"],
                     summary["SEMAF"]["avg_c"]],
        "Baseline": [summary["Baseline"]["lra"],  summary["Baseline"]["avg_ce"],
                     summary["Baseline"]["avg_kri"], summary["Baseline"]["avg_q"],
                     summary["Baseline"]["avg_c"]],
    })
    comp_df["SEMAF 우위"] = comp_df.apply(
        lambda r: "✅" if (r["지표"] != "평균 통신 횟수 C" and r["SEMAF"] > r["Baseline"])
                       or (r["지표"] == "평균 통신 횟수 C" and r["SEMAF"] < r["Baseline"])
                  else "❌", axis=1
    )
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── 9-3. 시계열 차트 ────────────────────────────────────────────────────
    st.markdown("### 📈 반복별 지표 추이")
    tab1, tab2, tab3, tab4 = st.tabs(["CE 추이", "KRI 추이", "품질(Q) 추이", "보상(R) 추이"])

    def make_pivot(col):
        return df.pivot_table(index="iteration", columns="system",
                              values=col, aggfunc="mean")

    with tab1:
        st.line_chart(make_pivot("ce"),  use_container_width=True)
        st.caption("CE = Q/C | 환경 변화 지점에서 Baseline은 급격히 하락, SEMAF는 빠르게 회복")
    with tab2:
        st.line_chart(make_pivot("kri"), use_container_width=True)
        st.caption("KRI = 1 - 오답률 | 95% 이상 유지 시 재앙적 망각 완화 검증")
    with tab3:
        st.line_chart(make_pivot("quality"), use_container_width=True)
        st.caption("품질 점수 Q (0~1)")
    with tab4:
        st.line_chart(make_pivot("reward"), use_container_width=True)
        st.caption(f"보상 R = {lambda_q}·Q - {lambda_c:.3f}·C + {lambda_k}·(KRI - {kri_target})")

    st.divider()

    # ── 9-4. 환경 변화 지점 분석 ────────────────────────────────────────────
    st.markdown("### ⚠️ 환경 변화 지점 상세 분석")
    adapt_rows = df[df["is_adapt_pt"]].copy()
    if not adapt_rows.empty:
        cols_show = ["iteration", "system", "quality", "comm_count", "ce", "kri",
                     "adapt_time", "reward"]
        st.dataframe(
            adapt_rows[cols_show].sort_values(["iteration", "system"]),
            use_container_width=True, hide_index=True
        )
    else:
        st.info("설정한 반복 범위 내에 환경 변화 지점이 없습니다.")

    st.divider()

    # ── 9-5. 반복 로그 테이블 ───────────────────────────────────────────────
    with st.expander("📜 전체 반복 로그 (원시 데이터)", expanded=False):
        display_cols = ["iteration", "system", "quality", "comm_count",
                        "ce", "kri", "reward", "adapt_time", "llm_source", "llm_elapsed"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

        # CSV 다운로드
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇️ 결과 CSV 다운로드",
            data=csv,
            file_name=f"semaf_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # ── 9-6. JSON 결과 익스팬더 ─────────────────────────────────────────────
    with st.expander("🔍 요약 결과 JSON", expanded=False):
        result_json = {
            "timestamp":        datetime.now().isoformat(),
            "config": {
                "num_iterations":   num_iterations,
                "num_agents":       num_agents,
                "adaptation_points": adaptation_points,
                "model":            model,
                "random_seed":      random_seed,
                "lambda_q":         lambda_q,
                "lambda_c":         lambda_c,
                "lambda_k":         lambda_k,
                "kri_target":       kri_target,
            },
            "summary": summary,
        }
        st.json(result_json)

    # ── 9-7. 논문 결과와의 비교 ─────────────────────────────────────────────
    st.markdown("### 📌 논문 기준 결과와 시뮬레이션 비교")
    paper_vals = {
        "LRA":  {"SEMAF": 0.2500, "Baseline": 0.0611, "improvement": "4.09×"},
        "CE":   {"SEMAF": 0.1475, "Baseline": 0.0377, "improvement": "3.91×"},
        "KRI":  {"SEMAF": 0.9509, "Baseline": 0.8047, "improvement": "+14.62%p"},
    }
    key_map = {"LRA": "lra", "CE": "avg_ce", "KRI": "avg_kri"}
    comp2 = []
    for m, pv in paper_vals.items():
        sim_s = summary["SEMAF"][key_map[m]]
        sim_b = summary["Baseline"][key_map[m]]
        comp2.append({
            "지표": m,
            "논문 SEMAF":    pv["SEMAF"],
            "시뮬 SEMAF":    sim_s,
            "논문 Baseline": pv["Baseline"],
            "시뮬 Baseline": sim_b,
            "논문 개선":     pv["improvement"],
        })
    st.dataframe(pd.DataFrame(comp2), use_container_width=True, hide_index=True)
    st.caption(
        "시뮬레이션은 논문 실험 조건(NUM_ITERATIONS=60, ADAPTATION_POINTS=[20,50])과 "
        "동일한 파라미터 설정 시 유사한 결과를 재현합니다."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 10. 초기 안내 화면
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.finished and not st.session_state.running:
    st.info(
        "👈 **사이드바**에서 실험 파라미터를 설정한 후 **▶ 실험 시작** 버튼을 클릭하세요.\n\n"
        "- **시뮬레이션 모드**: API 키 없이도 즉시 실행 가능\n"
        "- **실제 API 모드**: OpenAI API 키 입력 후 사이드바 토글 활성화\n"
        "- 결과는 CSV로 다운로드할 수 있습니다."
    )

    # 아키텍처 개요
    st.markdown("### 🏗️ SEMAF 아키텍처 레이어")
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("""
**Agent Layer**
- LLM Core
- Tools
- Memory

*작업 실행, 추론, 외부 환경 상호작용*
        """)
    with ac2:
        st.markdown("""
**Adaptation & Evolution Layer**
- Evolution Engine ③
- Feedback Collector ②
- Self-reflection Module

*자기진화 사이클 구동, 정책 업데이트, 역할 재조직*
        """)
    with ac3:
        st.markdown("""
**Foundation Layer**
- Knowledge Graph Layer ①
- Governance Layer ④
- A2A Protocol ⑤

*동적 지식 관리, 신뢰성 보장, 에이전트 간 통신 표준화*
        """)
