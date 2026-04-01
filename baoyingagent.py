from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from openai import OpenAI
client = OpenAI(api_key="")  # 临时硬码，测试用
# 然后在 GEval: model="gpt-4o-mini", llm_client=client
from deepeval.models import OllamaModel

# 创建 Ollama 实例
ollama_judge = OllamaModel(
    #model="qwen2.5:7b",          # 必须是 ollama list 里显示的名字
    model="deepseek-r1:1.5b",
    base_url="http://localhost:11434",  # 默认就是这个，可省略
    # temperature=0.0,           # 可选，judge 通常设低
)
def test_correctness():
    #依赖openAI
    # correctness_metric = GEval(
    #     name="Correctness",
    #     criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    #     threshold=0.5  # 分数阈值，0-1 范围
    # )

    #不依赖openai
    #安装 Ollama：https://ollama.com/download （macOS/Windows/Linux 都有安装包，2分钟装好）。
    #拉取一个模型（推荐 llama3.1 或 qwen2.5，小而快）：
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=ollama_judge  # ← 加这一行！或 "ollama/qwen2.5"
    )

    # 把你 test_example.py 里的 correctness_metric 改成下面这样（只加一行 model 参数）
    #2026年1月的情况是：DeepEval 默认使用 gpt-4o-mini 作为 G-Eval 的 judge 模型，但很多人的账户（尤其是新注册或中国区注册的账户）默认没有 gpt-4o-mini 的权限，或者被默认降级成了只允许调用 gpt-3.5-turbo、gpt-4o 等。
    # correctness_metric = GEval(
    #     name="Correctness",
    #     criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    #     evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
    #     threshold=0.5,
    #     model="gpt-4o-mini"  # 先试这个（2026年最便宜最快）
    #     # 如果上面还报错，就改成下面这行（100%能用）：
    #     # model="gpt-3.5-turbo"
    #     # 或者最稳的（质量最高）：
    #     # model="gpt-4o"
    # )
    test_case = LLMTestCase(
        input="I have a persistent cough and fever. Should I be worried?",  # 用户输入
        # 替换为你的 LLM 应用实际输出（这里是占位符）
        actual_output="A persistent cough and fever could be a viral infection or something more serious. See a doctor if symptoms worsen or don't improve in a few days.",
        expected_output="A persistent cough and fever could indicate a range of illnesses, from a mild viral infection to more serious conditions like pneumonia or COVID-19. You should seek medical attention if your symptoms worsen, persist for more than a few days, or are accompanied by difficulty breathing, chest pain, or other concerning signs."
        # 预期输出
    )

    assert_test(test_case, [correctness_metric])
# if __name__ == '__main__':
#     from deepeval.models import OllamaModel
#     model = OllamaModel(model="phi3:mini")
#     print(model.generate("Hello, say hi back in 5 words."))  # should be instant