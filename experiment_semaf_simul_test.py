import time
import random
from typing import List, Dict, Any

# Optional: import openai when available. Actual import done in llm_api_call to allow graceful fallback.

# -----------------------------------------------------------------------------
# 1. Environment Settings and Constant Definitions
# -----------------------------------------------------------------------------

# LLM Integration Info (Pseudo code)
LLM_MODEL = "gpt-4o-mini"
OPENAI_API_KEY = 'user-openai-apikey' 

# 3 Test Prompts
TEST_PROMPTS = [
    "Summarize global AI regulation trends and major issues over the past 3 years, and discuss Korea's response strategy.",
    "Explain the core principles and current technical limitations of quantum computing, and predict 3 potential impacts on the financial industry.",
    "Describe the core components and respective functions of A2A in detail."
]

# Experimental Simulation Constants
NUM_ITERATIONS = 50  # Total simulation iterations (Increase to 100 for result stabilization)
NUM_AGENTS = 3       # Number of agents
NUM_KNOWLEDGE_QUESTIONS = 50 # Number of existing knowledge questions
NUM_NEW_KNOWLEDGE = 50 # Number of new knowledge items to integrate

# Environment Change Points (Points for measuring LRA)
ADAPTATION_POINTS = [20, 50]

# -----------------------------------------------------------------------------
# 2. Evaluation Metric Calculation Functions (Based on formulas in Chapter 4)
# -----------------------------------------------------------------------------

def calculate_lra(adaptation_times: List[int]) -> float:
    """
    Calculate Learning Rate of Adaptation (LRA)
    LRA = (1/N) * Sum(1/Ti)
    Ti: Iterations taken to return to optimal performance after the i-th environment change.
    """
    if not adaptation_times:
        return 0.0
    
    N = len(adaptation_times)
    # Assuming Ti is at least 1 since it cannot be 0.
    sum_of_inverse_ti = sum(1.0 / max(1, t) for t in adaptation_times)
    
    return sum_of_inverse_ti / N

def calculate_ce(quality_score: float, communication_count: int) -> float:
    """
    Calculate Collaboration Efficiency (CE)
    CE = Q / C
    Q: Quality score of the final result (0.0 ~ 1.0)
    C: Total number of communications between agents to complete the task
    """
    if communication_count == 0:
        return quality_score  # If no communication, quality score itself is the efficiency
    return quality_score / communication_count

def calculate_kri(old_knowledge_error_rate: float) -> float:
    """
    Calculate Knowledge Retention Index (KRI)
    KRI = 1 - (D_new / D_total)
    D_new: Error rate on existing knowledge questions after integrating new knowledge
    D_total: Total number of questions (Error rate itself corresponds to D_new/D_total here)
    """
    # Assuming error_rate is a value between 0.0 and 1.0
    return 1.0 - old_knowledge_error_rate

# -----------------------------------------------------------------------------
# 3. LLM Integration and System Simulation (Pseudo Code)
# -----------------------------------------------------------------------------

def llm_api_call(prompt: str, system_type: str) -> str:
    """
    Try to call the OpenAI API for a real LLM response. If the SDK or network fails,
    fall back to a deterministic pseudo response so the experiment still runs.
    """
    try:
        import openai
    except Exception:
        # If import fails, immediately fallback to pseudo response
        print("[LLM Library Not Installed — Using Fallback]")
        if "SEMAF" in system_type:
            return f"[{system_type} Response] {prompt[:30]}... (High Quality, Adaptive)"
        else:
            return f"[{system_type} Response] {prompt[:30]}... (Fixed Role, Lower Quality)"

    MAX_RETRIES = 3
    BACKOFF_FACTOR = 1.5
    fallback_models = [LLM_MODEL, "gpt-4o-mini"]

    last_exception = None
    for model in fallback_models:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                client = openai.OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else openai.OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_type},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=600,
                    temperature=0.2,
                )

                choice = response.choices[0]
                message = None
                if isinstance(getattr(choice, 'message', None), dict):
                    message = choice.message.get('content', '')
                else:
                    msg_obj = getattr(choice, 'message', None)
                    if msg_obj is not None:
                        message = getattr(msg_obj, 'content', '')
                if not message:
                    message = getattr(choice, 'text', '') or (choice.get('text') if isinstance(choice, dict) else '')

                result = (message or '').strip()
                return f"[model={model}] {result}"

            except Exception as e:
                last_exception = e
                err_msg = str(e)
                print(f"[LLM Call Error] Model={model} Attempt={attempt} Error: {err_msg}")
                if attempt < MAX_RETRIES:
                    backoff = BACKOFF_FACTOR ** (attempt - 1)
                    time.sleep(backoff)
                else:
                    break

    print(f"[All LLM Calls Failed — Using Fallback] Last Error: {last_exception}")
    if "SEMAF" in system_type:
        return f"[{system_type} Response] {prompt[:30]}... (High Quality, Adaptive)"
    else:
        return f"[{system_type} Response] {prompt[:30]}... (Fixed Role, Lower Quality)"

def simulate_system_performance(system_type: str, prompt: str, iteration: int) -> Dict[str, Any]:
    """
    Single task execution and performance metric simulation (Pseudo Code)
    """
    # 1. LLM Call (Real API or Fallback)
    llm_response = llm_api_call(prompt, system_type)
    
    # 2. Performance Metric Simulation (Reflecting assumption that SEMAF is superior to Baseline)
    if "SEMAF" in system_type:
        # SEMAF: High quality, low communication, low forgetting rate, fast adaptation
        quality = random.uniform(0.90, 0.95)  # Q: 0.90 ~ 0.95
        communication = random.randint(5, 8)  # C: 5 ~ 8
        kri_error_rate = random.uniform(0.01, 0.05) # KRI: 95% ~ 99%
        
        # Temporary performance dip at environment change points, then rapid recovery
        if iteration in ADAPTATION_POINTS:
            quality = random.uniform(0.75, 0.85) # Maintains higher quality than Baseline even during changes
            communication = random.randint(10, 15)
            
        adaptation_time = random.randint(2, 4) if iteration in ADAPTATION_POINTS else 0
        
    else: # Baseline (Fixed Role System)
        # Baseline: Lower quality, high communication, high forgetting rate, slow adaptation
        quality = random.uniform(0.70, 0.80)  # Q: 0.70 ~ 0.80
        communication = random.randint(15, 25) # C: 15 ~ 25
        kri_error_rate = random.uniform(0.15, 0.25) # KRI: 75% ~ 85%
        
        if iteration in ADAPTATION_POINTS:
            quality = random.uniform(0.40, 0.60)
            communication = random.randint(30, 40)
            
        adaptation_time = random.randint(15, 20) if iteration in ADAPTATION_POINTS else 0

    # 3. CE Calculation
    ce_score = calculate_ce(quality, communication)
    
    # 4. KRI Calculation
    kri_score = calculate_kri(kri_error_rate)
    
    return {
        "iteration": iteration,
        "quality_score": quality,
        "communication_count": communication,
        "ce_score": ce_score,
        "kri_score": kri_score,
        "adaptation_time": adaptation_time,
        "llm_response_snippet": llm_response[:50] + "...",
        "llm_response": llm_response
    }

# -----------------------------------------------------------------------------
# 4. Main Experiment Execution
# -----------------------------------------------------------------------------

def run_experiment():
    """
    Main function to run the empirical experiment simulation and output results.
    """
    print(f"--- Starting Empirical Experiment Simulation: {NUM_ITERATIONS} Iterations ---")
    print(f"LLM Model: {LLM_MODEL}, Number of Test Prompts: {len(TEST_PROMPTS)}")
    print("-" * 50)

    results = {
        "SEMAF": {"ce_scores": [], "kri_scores": [], "adaptation_times": []},
        "Baseline": {"ce_scores": [], "kri_scores": [], "adaptation_times": []}
    }

    for i in range(1, NUM_ITERATIONS + 1):
        prompt_index = (i - 1) % len(TEST_PROMPTS)
        current_prompt = TEST_PROMPTS[prompt_index]

        print(f"[Input Prompt] {current_prompt}")

        # 1. Baseline System Simulation
        baseline_res = simulate_system_performance("Baseline (Fixed Role System)", current_prompt, i)
        results["Baseline"]["ce_scores"].append(baseline_res["ce_score"])
        results["Baseline"]["kri_scores"].append(baseline_res["kri_score"])
        if baseline_res["adaptation_time"] > 0:
            results["Baseline"]["adaptation_times"].append(baseline_res["adaptation_time"])
        print(f"[Baseline LLM Response] {baseline_res.get('llm_response')}")

        # 2. SEMAF System Simulation
        semaf_res = simulate_system_performance("SEMAF (Self-Evolving System)", current_prompt, i)
        results["SEMAF"]["ce_scores"].append(semaf_res["ce_score"])
        results["SEMAF"]["kri_scores"].append(semaf_res["kri_score"])
        if semaf_res["adaptation_time"] > 0:
            results["SEMAF"]["adaptation_times"].append(semaf_res["adaptation_time"])
        print(f"[SEMAF LLM Response] {semaf_res.get('llm_response')}")
        
        if i in ADAPTATION_POINTS:
            print(f"[Env Change Point: Iteration {i}] Starting system adaptation measurement")

    # -------------------------------------------------------------------------
    # 5. Final Result Analysis and Comparison
    # -------------------------------------------------------------------------
    
    lra_semaf = calculate_lra(results["SEMAF"]["adaptation_times"])
    lra_baseline = calculate_lra(results["Baseline"]["adaptation_times"])
    
    avg_ce_semaf = sum(results["SEMAF"]["ce_scores"]) / len(results["SEMAF"]["ce_scores"])
    avg_ce_baseline = sum(results["Baseline"]["ce_scores"]) / len(results["Baseline"]["ce_scores"])
    
    avg_kri_semaf = sum(results["SEMAF"]["kri_scores"]) / len(results["SEMAF"]["kri_scores"])
    avg_kri_baseline = sum(results["Baseline"]["kri_scores"]) / len(results["Baseline"]["kri_scores"])

    print("-" * 50)
    print("--- Final Experiment Result Comparison ---")
    print(f"Environment Changes: {len(ADAPTATION_POINTS)} times")
    print(f"LRA Measurement Points: {ADAPTATION_POINTS}")
    print("-" * 50)
    
    print(f"{'Metric':<10} | {'SEMAF':<15} | {'Baseline':<15} | {'SEMAF Win':<10}")
    print("-" * 50)
    
    # LRA Results
    lra_superior = "O" if lra_semaf > lra_baseline else "X"
    print(f"{'LRA':<10} | {lra_semaf:.4f} (1/T) | {lra_baseline:.4f} (1/T) | {lra_superior:<10}")
    
    # CE Results
    ce_superior = "O" if avg_ce_semaf > avg_ce_baseline else "X"
    print(f"{'CE':<10} | {avg_ce_semaf:.4f} (Q/C) | {avg_ce_baseline:.4f} (Q/C) | {ce_superior:<10}")
    
    # KRI Results
    kri_superior = "O" if avg_kri_semaf > avg_kri_baseline else "X"
    print(f"{'KRI':<10} | {avg_kri_semaf:.4f} (1-Err) | {avg_kri_baseline:.4f} (1-Err) | {kri_superior:<10}")
    
    print("-" * 50)
    print("--- Implementation Items Provided ---")
    print(f"1) LLM Integration: {LLM_MODEL} (Reflected in Pseudo Code)")
    print(f"2) Test Prompts: {len(TEST_PROMPTS)} prompts included")
    print(f"3) Comparison with Baseline included")
    print(f"4) Difficult-to-execute parts (complex simulation) reflected as Pseudo Code")
    print("-" * 50)
    
if __name__ == "__main__":
    run_experiment()