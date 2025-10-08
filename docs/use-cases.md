---
layout: default
title: Use Cases - LoRA Craft
---

# Use Cases

Explore real-world applications for fine-tuned language models with LoRA Craft.

---

## Math & Science Education {#math-science}

Train models to solve mathematical problems, explain scientific concepts, and provide step-by-step reasoning.

### Example Application: Math Tutoring Assistant

**The Challenge:**
Students need help with math homework but want detailed explanations, not just answers. Generic LLMs often skip steps or make calculation errors.

**The Solution:**
Fine-tune a model on math problem datasets with a reward function that values:
- Correct final answers
- Step-by-step reasoning
- Clear explanations of concepts

### Recommended Configuration

**Model:** Qwen3 1.5B or Llama 3.2 3B
- Fast inference for interactive tutoring
- Strong baseline math capabilities

**Dataset Options:**
- **GSM8K** (8.5K problems): Grade school math word problems
- **OpenMath** (100K problems): Advanced mathematical reasoning
- **Orca Math** (200K problems): Diverse difficulty levels

**Reward Function:** Math & Science
- Validates numerical accuracy
- Rewards showing work
- Penalizes calculation errors

**Training Tips:**
- Use pre-training to teach output format (reasoning → answer)
- Enable chain-of-thought markers in system prompt
- Train for 3-5 epochs on full dataset for production use

### Expected Results

**Before Fine-tuning:**
```
Q: If a shirt costs $25 and is on sale for 20% off, what's the final price?
A: The final price would be around $20.
```

**After Fine-tuning:**
```
Q: If a shirt costs $25 and is on sale for 20% off, what's the final price?

A: <start_working_out>
Original price: $25
Discount: 20% of $25 = 0.20 × $25 = $5
Final price: $25 - $5 = $20
<end_working_out>

<SOLUTION>$20</SOLUTION>
```

### Real-World Applications

- **Homework help chatbots**: Interactive tutoring for students
- **STEM education platforms**: Automated problem explanations
- **Test preparation tools**: Practice with instant feedback
- **Research assistants**: Scientific calculation verification

---

## Code Generation & Programming {#coding}

Create AI coding assistants that generate clean, efficient code with proper documentation.

### Example Application: Python Code Assistant

**The Challenge:**
Developers need code snippets that are not only syntactically correct but also follow best practices, include error handling, and are well-documented.

**The Solution:**
Fine-tune on code datasets with rewards for:
- Syntactic correctness
- PEP 8 style compliance
- Proper error handling
- Inline documentation

### Recommended Configuration

**Model:** Qwen3 3B or Mistral 7B
- Larger context for complex code
- Better instruction following

**Dataset Options:**
- **Code Alpaca** (20K examples): General programming tasks
- **Custom dataset**: Your specific codebase patterns
- **GitHub repos**: Domain-specific code examples

**Reward Function:** Code Generation or Programming
- Validates syntax via AST parsing
- Checks for common anti-patterns
- Rewards documentation and type hints

**Training Tips:**
- Include diverse programming tasks (CRUD, algorithms, data processing)
- Use temperature 0.3-0.5 for more deterministic outputs
- Test on held-out code problems for validation

### Expected Results

**Before Fine-tuning:**
```python
def read_file(path):
    f = open(path)
    data = f.read()
    return data
```

**After Fine-tuning:**
```python
def read_file(path: str) -> str:
    """
    Read and return the contents of a file.

    Args:
        path: Path to the file to read

    Returns:
        File contents as a string

    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except IOError as e:
        raise IOError(f"Error reading file {path}: {e}")
```

### Real-World Applications

- **IDE assistants**: Code completion and generation
- **Code review tools**: Suggest improvements and best practices
- **Documentation generators**: Auto-generate docstrings
- **Boilerplate creators**: Generate standard project structures
- **Bug fixers**: Identify and correct common errors

---

## Question Answering Systems {#qa}

Build specialized Q&A models for specific domains with accurate, relevant answers.

### Example Application: Customer Support Bot

**The Challenge:**
Customer support teams need to answer repetitive questions quickly while maintaining accuracy and helpful tone.

**The Solution:**
Fine-tune on your FAQ data and support tickets with rewards for:
- Answer accuracy
- Relevant information inclusion
- Concise, helpful responses
- Professional tone

### Recommended Configuration

**Model:** Llama 3.2 3B or Qwen3 1.5B
- Balance between quality and inference speed
- Good for high-volume requests

**Dataset Options:**
- **SQuAD v2** (130K questions): Reading comprehension baseline
- **Custom FAQ dataset**: Your company's actual questions
- **Support ticket history**: Real customer interactions

**Reward Function:** Question Answering
- Validates factual accuracy
- Rewards relevant details
- Penalizes hallucinations

**Training Tips:**
- Include "I don't know" examples for unanswerable questions
- Use domain-specific terminology in system prompt
- Fine-tune on actual customer question patterns

### Expected Results

**Before Fine-tuning:**
```
Q: What's your return policy?
A: You can return items. Check our website for more details.
```

**After Fine-tuning:**
```
Q: What's your return policy?
A: We offer a 30-day return window for unused items in original packaging.
To initiate a return:
1. Log into your account
2. Go to Order History
3. Select the item and click "Return"
4. Print the prepaid shipping label

Refunds are processed within 5-7 business days of receiving the return.
For damaged or defective items, returns are accepted beyond 30 days.
```

### Real-World Applications

- **Customer support automation**: Reduce ticket volume
- **Knowledge base assistants**: Internal company Q&A
- **Medical Q&A**: Patient information systems (with medical datasets)
- **Legal research**: Case law and regulation queries
- **Technical documentation**: Developer help systems

---

## Custom Domain Applications {#custom}

Fine-tune for specialized tasks with custom reward functions and datasets.

### Example 1: Medical Report Summarization

**Application:** Summarize lengthy medical reports for doctors.

**Configuration:**
- **Model:** Llama 3.2 3B
- **Dataset:** Medical reports with human-written summaries
- **Reward:** Concise Summarization + Medical Accuracy
- **Special considerations:** HIPAA compliance, medical terminology

**Key Metrics:**
- Summary conciseness (target: 20% of original length)
- Retention of critical findings
- Proper medical terminology usage

### Example 2: Legal Document Analysis

**Application:** Extract key clauses and obligations from contracts.

**Configuration:**
- **Model:** Mistral 7B (large context window)
- **Dataset:** Annotated legal contracts
- **Reward:** Custom legal extraction reward
- **Special considerations:** Precision over recall

**Key Metrics:**
- Clause identification accuracy
- Obligation extraction completeness
- Legal terminology precision

### Example 3: Creative Content Generation

**Application:** Generate marketing copy with specific brand voice.

**Configuration:**
- **Model:** Qwen3 3B
- **Dataset:** Approved marketing materials
- **Reward:** Creative Writing + Brand Consistency
- **Special considerations:** Tone matching, keyword inclusion

**Key Metrics:**
- Brand voice consistency score
- Engagement predicted metrics
- SEO keyword integration

### Example 4: Language Translation

**Application:** Domain-specific translation (technical, medical, legal).

**Configuration:**
- **Model:** Qwen3 3B or larger
- **Dataset:** Parallel corpus in your domain
- **Reward:** Custom translation quality metric (BLEU + domain terms)
- **Special considerations:** Preserve technical terms, cultural adaptation

**Key Metrics:**
- BLEU/METEOR scores
- Domain terminology accuracy
- Fluency in target language

---

## Building Custom Reward Functions

For specialized applications, you'll need custom reward functions. Here's how:

### 1. Define Success Criteria

What makes a "good" output for your task?
- Specific format requirements?
- Factual accuracy constraints?
- Style or tone preferences?
- Length limitations?

### 2. Implement Reward Logic

Example: Product description generator
```python
def product_description_reward(response, reference_data):
    score = 0.0

    # Check length (100-200 words ideal)
    word_count = len(response.split())
    if 100 <= word_count <= 200:
        score += 0.3

    # Check for required elements
    required_elements = ['features', 'benefits', 'use case']
    for element in required_elements:
        if element.lower() in response.lower():
            score += 0.2

    # Check sentiment (should be positive)
    sentiment = analyze_sentiment(response)  # Custom function
    if sentiment > 0.5:
        score += 0.3

    return min(score, 1.0)
```

### 3. Test Thoroughly

Before training:
- Test reward on 20+ diverse examples
- Verify score distribution (not all 0.0 or 1.0)
- Check edge cases
- Compare with human judgments

### 4. Iterate Based on Results

After training:
- Review model outputs
- Adjust reward weights
- Add new criteria if needed
- Re-train with updated reward

---

## Getting Started with Your Use Case

### 1. Define Your Task

- What inputs does the model receive?
- What outputs should it produce?
- What makes an output "good"?

### 2. Gather Data

- Minimum 500 examples for initial testing
- 2,000+ for production deployment
- Include diverse examples and edge cases

### 3. Choose Starting Point

- Similar use case from this page
- Closest pre-built reward function
- Model size based on complexity

### 4. Iterate Quickly

- Start with small subset (100-500 samples)
- Train for 1-2 epochs
- Evaluate and adjust
- Scale up when satisfied

### 5. Measure Success

Define metrics before training:
- Accuracy/F1 for classification
- BLEU/ROUGE for generation
- Task-specific metrics
- Human evaluation criteria

---

## Need Help with Your Use Case?

- **Share in Discussions**: [GitHub Discussions](https://github.com/jwest33/lora_craft/discussions)
- **Request Features**: [GitHub Issues](https://github.com/jwest33/lora_craft/issues)
- **Read the Docs**: [Technical Reference](documentation.html)

Have a success story? We'd love to hear it! Share your results in our community discussions.
