"""Combined Model and Training configuration tab."""

import tkinter as tk
from tkinter import ttk
import webbrowser
from typing import Callable, Dict, Any
try:
    from ..styled_widgets import (
        StyledFrame, StyledLabelFrame, StyledButton,
        StyledEntry, StyledLabel, SectionHeader,
        FormRow, ButtonGroup, Card
    )
    from ..style_constants import FONTS, SPACING, PADDING, COLORS
    STYLED_WIDGETS = True
except ImportError:
    STYLED_WIDGETS = False


class ModelTrainingTab:
    """Combined tab for model configuration and training settings."""

    def __init__(self, parent, on_config_change: Callable):
        """Initialize model and training tab."""
        self.on_config_change = on_config_change
        self.frame = ttk.Frame(parent)
        self._create_widgets()

    def _create_widgets(self):
        """Create tab widgets."""
        # Main container with scrollbar
        canvas = tk.Canvas(self.frame)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Bind mouse wheel for scrolling
        def _on_mousewheel(event):
            try:
                widget = self.frame.winfo_containing(event.x_root, event.y_root)
                if widget:
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            except:
                pass

        def _bind_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")

        scrollable_frame.bind("<Enter>", _bind_mousewheel)
        scrollable_frame.bind("<Leave>", _unbind_mousewheel)

        # Documentation link at the top with enhanced styling
        doc_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
        doc_frame.pack(fill=tk.X, padx=SPACING['lg'] if STYLED_WIDGETS else 10,
                      pady=(0, SPACING['lg'] if STYLED_WIDGETS else 10))

        if STYLED_WIDGETS:
            doc_button = StyledButton(
                doc_frame,
                text="LoRA Hyperparameters Documentation",
                command=lambda: webbrowser.open("https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide"),
                style_type='primary'
            )
            doc_button.pack(side=tk.LEFT, padx=SPACING['md'], pady=SPACING['sm'])

        else:
            doc_button = ttk.Button(
                doc_frame,
                text="Unsloth LoRA Hyperparameters Documentation",
                command=lambda: webbrowser.open("https://docs.unsloth.ai/get-started/fine-tuning-llms-guide/lora-hyperparameters-guide")
            )
            doc_button.pack(side=tk.LEFT, padx=5)

        # Model Selection Section with enhanced styling
        if STYLED_WIDGETS:
            # Add section header
            SectionHeader(scrollable_frame, "Model Selection",
                         "Choose the base model and configure LoRA parameters").pack(
                fill=tk.X, padx=SPACING['lg'], pady=(0, SPACING['md']))

            model_frame = Card(scrollable_frame, padding='section')
            model_content = model_frame.content
        else:
            model_frame = ttk.LabelFrame(scrollable_frame, text="Model Selection", padding=15)
            model_content = model_frame

        model_frame.pack(fill=tk.X, padx=SPACING['lg'] if STYLED_WIDGETS else 10,
                        pady=SPACING['md'] if STYLED_WIDGETS else 5)
        model_content.columnconfigure(1, weight=1)

        # Model selection with enhanced label
        if STYLED_WIDGETS:
            StyledLabel(model_content, "Model:", style_type='label_bold').grid(
                row=0, column=0, sticky=tk.W, padx=(SPACING['sm'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(model_content, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.model_name = tk.StringVar(value="qwen2.5-0.5b")
        models = [
            "qwen2.5-0.5b", "qwen2.5-1.5b", "qwen2.5-3b", "qwen2.5-7b",
            "qwen2.5-14b", "qwen2.5-32b",
            "llama-3.2-1b", "llama-3.2-3b", "llama-3.1-8b", "llama-3.1-70b",
            "mistral-7b", "phi-3-mini", "gemma-2b", "gemma-7b"
        ]
        model_combo = ttk.Combobox(model_content, textvariable=self.model_name, values=models,
                                   width=35, style='Styled.TCombobox' if STYLED_WIDGETS else '')
        model_combo.grid(row=0, column=1, sticky=tk.W,
                        padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                        pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        # Model info label with enhanced styling
        if STYLED_WIDGETS:
            self.model_info = StyledLabel(model_content, "", style_type='small_italic')
            self.model_info.configure(foreground='gray')
        else:
            self.model_info = ttk.Label(model_content, text="", font=('TkDefaultFont', 9, 'italic'))
        self.model_info.grid(row=0, column=2, padx=SPACING['md'] if STYLED_WIDGETS else 10,
                           pady=SPACING['sm'] if STYLED_WIDGETS else 5)
        model_combo.bind("<<ComboboxSelected>>", self._update_model_info)

        # LoRA Configuration Section with enhanced styling
        if STYLED_WIDGETS:
            SectionHeader(scrollable_frame, "LoRA Configuration",
                         "Fine-tuning adapter settings").pack(
                fill=tk.X, padx=SPACING['lg'], pady=(SPACING['lg'], SPACING['md']))

            lora_frame = Card(scrollable_frame, padding='section')
            lora_content = lora_frame.content
        else:
            lora_frame = ttk.LabelFrame(scrollable_frame, text="LoRA Configuration", padding=15)
            lora_content = lora_frame

        lora_frame.pack(fill=tk.X, padx=SPACING['lg'] if STYLED_WIDGETS else 10,
                       pady=SPACING['md'] if STYLED_WIDGETS else 5)
        lora_content.columnconfigure(1, weight=1)
        lora_content.columnconfigure(3, weight=1)

        # LoRA Rank with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(lora_content, "LoRA Rank:", style_type='label_bold').grid(
                row=0, column=0, sticky=tk.W, padx=(SPACING['sm'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(lora_content, text="LoRA Rank:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)

        self.lora_rank = tk.IntVar(value=16)
        rank_spinbox = ttk.Spinbox(lora_content, from_=1, to=256, textvariable=self.lora_rank, width=15,
                                   style='Styled.TSpinbox' if STYLED_WIDGETS else '')
        rank_spinbox.grid(row=0, column=1, sticky=tk.W,
                         padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                         pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        if STYLED_WIDGETS:
            hint_label = StyledLabel(lora_content, "Higher = more parameters", style_type='small_italic')
            hint_label.configure(foreground='gray')
        else:
            hint_label = ttk.Label(lora_content, text="Higher = more parameters",
                                  font=('TkDefaultFont', 9, 'italic'))
        hint_label.grid(row=0, column=2, padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                       pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        # LoRA Alpha
        ttk.Label(lora_content, text="LoRA Alpha:").grid(row=0, column=3, sticky=tk.W, padx=(20, 15), pady=5)
        self.lora_alpha = tk.IntVar(value=16)
        alpha_spinbox = ttk.Spinbox(lora_content, from_=1, to=256, textvariable=self.lora_alpha, width=15)
        alpha_spinbox.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(lora_content, text="Usually equals rank",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=5, padx=5, pady=5)

        # LoRA Dropout
        ttk.Label(lora_content, text="LoRA Dropout:").grid(row=1, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.lora_dropout = tk.DoubleVar(value=0.1)
        dropout_spinbox = ttk.Spinbox(lora_content, from_=0.0, to=0.5, increment=0.05,
                                     textvariable=self.lora_dropout, width=15)
        dropout_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(lora_content, text="Regularization",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=1, column=2, padx=5, pady=5)

        # Target Modules
        ttk.Label(lora_content, text="Target Modules:").grid(row=1, column=3, sticky=tk.W, padx=(20, 15), pady=5)
        self.target_modules = tk.StringVar(value="q_proj,v_proj,k_proj,o_proj")
        modules_entry = ttk.Entry(lora_content, textvariable=self.target_modules, width=25)
        modules_entry.grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(lora_content, text="Comma-separated",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=1, column=5, padx=5, pady=5)

        # Training Parameters Section with enhanced styling
        if STYLED_WIDGETS:
            SectionHeader(scrollable_frame, "Training Parameters",
                         "Core training configuration").pack(
                fill=tk.X, padx=SPACING['lg'], pady=(SPACING['lg'], SPACING['md']))

            training_frame = Card(scrollable_frame, padding='section')
            training_content = training_frame.content
        else:
            training_frame = ttk.LabelFrame(scrollable_frame, text="Training Parameters", padding=15)
            training_content = training_frame

        training_frame.pack(fill=tk.X, padx=SPACING['lg'] if STYLED_WIDGETS else 10,
                           pady=SPACING['md'] if STYLED_WIDGETS else 5)
        training_content.columnconfigure(1, weight=1)
        training_content.columnconfigure(3, weight=1)

        # Learning Rate with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(training_content, "Learning Rate:", style_type='label_bold').grid(
                row=0, column=0, sticky=tk.W, padx=(SPACING['sm'], SPACING['lg']), pady=SPACING['sm'])
            self.learning_rate = tk.StringVar(value="2e-4")
            lr_entry = StyledEntry(training_content, textvariable=self.learning_rate, placeholder="e.g., 2e-4")
            lr_entry.configure(width=15)
        else:
            ttk.Label(training_content, text="Learning Rate:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)
            self.learning_rate = tk.StringVar(value="2e-4")
            lr_entry = ttk.Entry(training_content, textvariable=self.learning_rate, width=15)

        lr_entry.grid(row=0, column=1, sticky=tk.W,
                     padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                     pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        # Batch Size
        ttk.Label(training_content, text="Batch Size:").grid(row=0, column=2, sticky=tk.W, padx=(20, 15), pady=5)
        self.batch_size = tk.IntVar(value=4)
        batch_spinbox = ttk.Spinbox(training_content, from_=1, to=128, textvariable=self.batch_size, width=15)
        batch_spinbox.grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)

        # Gradient Accumulation
        ttk.Label(training_content, text="Gradient Accumulation:").grid(row=1, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.gradient_accumulation = tk.IntVar(value=1)
        grad_spinbox = ttk.Spinbox(training_content, from_=1, to=32, textvariable=self.gradient_accumulation, width=15)
        grad_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        # Epochs
        ttk.Label(training_content, text="Epochs:").grid(row=1, column=2, sticky=tk.W, padx=(20, 15), pady=5)
        self.num_epochs = tk.IntVar(value=3)
        epoch_spinbox = ttk.Spinbox(training_content, from_=1, to=100, textvariable=self.num_epochs, width=15)
        epoch_spinbox.grid(row=1, column=3, sticky=tk.W, padx=5, pady=5)

        # Warmup Steps
        ttk.Label(training_content, text="Warmup Steps:").grid(row=2, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.warmup_steps = tk.IntVar(value=100)
        warmup_spinbox = ttk.Spinbox(training_content, from_=0, to=5000, textvariable=self.warmup_steps, width=15)
        warmup_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        # Max Steps
        ttk.Label(training_content, text="Max Steps (-1 = auto):").grid(row=2, column=2, sticky=tk.W, padx=(20, 15), pady=5)
        self.max_steps = tk.IntVar(value=-1)
        steps_spinbox = ttk.Spinbox(training_content, from_=-1, to=100000, textvariable=self.max_steps, width=15)
        steps_spinbox.grid(row=2, column=3, sticky=tk.W, padx=5, pady=5)

        # GRPO Loss Configuration Section
        grpo_frame = ttk.LabelFrame(scrollable_frame, text="GRPO Loss Configuration", padding=15)
        grpo_frame.pack(fill=tk.X, padx=10, pady=5)
        grpo_frame.columnconfigure(1, weight=1)
        grpo_frame.columnconfigure(3, weight=1)

        # Loss Type Selection
        ttk.Label(grpo_frame, text="Loss Type:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.loss_type = tk.StringVar(value="grpo")
        loss_types = ["grpo", "gspo", "dr_grpo"]
        loss_combo = ttk.Combobox(grpo_frame, textvariable=self.loss_type, values=loss_types, width=15, state="readonly")
        loss_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        loss_combo.bind("<<ComboboxSelected>>", self._on_loss_type_change)

        # Loss type description
        self.loss_info = ttk.Label(grpo_frame, text="Standard GRPO", font=('TkDefaultFont', 9, 'italic'))
        self.loss_info.grid(row=0, column=2, columnspan=3, sticky=tk.W, padx=10, pady=5)

        # Mask Truncated Completions
        self.mask_truncated = tk.BooleanVar(value=True)
        mask_cb = ttk.Checkbutton(grpo_frame, text="Mask Truncated Completions",
                                 variable=self.mask_truncated)
        mask_cb.grid(row=1, column=0, columnspan=2, sticky=tk.W, padx=(5, 15), pady=10)

        # Dynamic parameters frame for loss-specific settings
        self.dynamic_frame = ttk.Frame(grpo_frame)
        self.dynamic_frame.grid(row=2, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=5)

        # Initialize with GRPO parameters
        self._create_grpo_params()

        # Generation Parameters Section with enhanced styling
        if STYLED_WIDGETS:
            SectionHeader(scrollable_frame, "Generation Parameters",
                         "Configure text generation settings").pack(
                fill=tk.X, padx=SPACING['lg'], pady=(SPACING['lg'], SPACING['md']))

            gen_frame = Card(scrollable_frame, padding='section')
            gen_content = gen_frame.content
        else:
            gen_frame = ttk.LabelFrame(scrollable_frame, text="Generation Parameters", padding=15)
            gen_content = gen_frame

        gen_frame.pack(fill=tk.X, padx=SPACING['lg'] if STYLED_WIDGETS else 10,
                      pady=SPACING['md'] if STYLED_WIDGETS else 5)
        gen_content.columnconfigure(1, weight=1)
        gen_content.columnconfigure(3, weight=1)

        # Temperature with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(gen_content, "Temperature:", style_type='label_bold').grid(
                row=0, column=0, sticky=tk.W, padx=(SPACING['sm'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(gen_content, text="Temperature:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)

        self.temperature = tk.DoubleVar(value=0.7)
        temp_scale = ttk.Scale(gen_content, from_=0.1, to=2.0, variable=self.temperature,
                              orient=tk.HORIZONTAL, length=150,
                              style='Styled.Horizontal.TScale' if STYLED_WIDGETS else '')
        temp_scale.grid(row=0, column=1, sticky=tk.W,
                       padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                       pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        if STYLED_WIDGETS:
            self.temp_label = StyledLabel(gen_content, f"{self.temperature.get():.2f}", style_type='value')
        else:
            self.temp_label = ttk.Label(gen_content, text=f"{self.temperature.get():.2f}")
        self.temp_label.grid(row=0, column=2,
                           padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                           pady=SPACING['sm'] if STYLED_WIDGETS else 5)
        temp_scale.configure(command=lambda v: self.temp_label.config(text=f"{float(v):.2f}"))

        # Top-p with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(gen_content, "Top-p:", style_type='label_bold').grid(
                row=0, column=3, sticky=tk.W, padx=(SPACING['xl'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(gen_content, text="Top-p:").grid(row=0, column=3, sticky=tk.W, padx=(20, 15), pady=5)

        self.top_p = tk.DoubleVar(value=0.95)
        top_p_scale = ttk.Scale(gen_content, from_=0.1, to=1.0, variable=self.top_p,
                               orient=tk.HORIZONTAL, length=150,
                               style='Styled.Horizontal.TScale' if STYLED_WIDGETS else '')
        top_p_scale.grid(row=0, column=4, sticky=tk.W,
                        padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                        pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        if STYLED_WIDGETS:
            self.top_p_label = StyledLabel(gen_content, f"{self.top_p.get():.2f}", style_type='value')
        else:
            self.top_p_label = ttk.Label(gen_content, text=f"{self.top_p.get():.2f}")
        self.top_p_label.grid(row=0, column=5,
                            padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                            pady=SPACING['sm'] if STYLED_WIDGETS else 5)
        top_p_scale.configure(command=lambda v: self.top_p_label.config(text=f"{float(v):.2f}"))

        # Top-k with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(gen_content, "Top-k:", style_type='label_bold').grid(
                row=1, column=0, sticky=tk.W, padx=(SPACING['sm'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(gen_content, text="Top-k:").grid(row=1, column=0, sticky=tk.W, padx=(5, 15), pady=5)

        self.top_k = tk.IntVar(value=50)
        top_k_spinbox = ttk.Spinbox(gen_content, from_=1, to=100, textvariable=self.top_k, width=15,
                                    style='Styled.TSpinbox' if STYLED_WIDGETS else '')
        top_k_spinbox.grid(row=1, column=1, sticky=tk.W,
                          padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                          pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        # Generations per Prompt with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(gen_content, "Generations per Prompt:", style_type='label_bold').grid(
                row=1, column=3, sticky=tk.W, padx=(SPACING['xl'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(gen_content, text="Generations per Prompt:").grid(row=1, column=3, sticky=tk.W, padx=(20, 15), pady=5)

        self.num_generations = tk.IntVar(value=4)
        gen_spinbox = ttk.Spinbox(gen_content, from_=1, to=10, textvariable=self.num_generations, width=15,
                                  style='Styled.TSpinbox' if STYLED_WIDGETS else '')
        gen_spinbox.grid(row=1, column=4, sticky=tk.W,
                        padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                        pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        # Max New Tokens with enhanced styling
        if STYLED_WIDGETS:
            StyledLabel(gen_content, "Max New Tokens:", style_type='label_bold').grid(
                row=2, column=0, sticky=tk.W, padx=(SPACING['sm'], SPACING['lg']), pady=SPACING['sm'])
        else:
            ttk.Label(gen_content, text="Max New Tokens:").grid(row=2, column=0, sticky=tk.W, padx=(5, 15), pady=5)

        self.max_new_tokens = tk.IntVar(value=512)
        tokens_spinbox = ttk.Spinbox(gen_content, from_=32, to=4096, increment=32,
                                    textvariable=self.max_new_tokens, width=15,
                                    style='Styled.TSpinbox' if STYLED_WIDGETS else '')
        tokens_spinbox.grid(row=2, column=1, sticky=tk.W,
                          padx=SPACING['sm'] if STYLED_WIDGETS else 5,
                          pady=SPACING['sm'] if STYLED_WIDGETS else 5)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _update_model_info(self, event=None):
        """Update model info label based on selection."""
        model = self.model_name.get()
        model_info = {
            "qwen2.5-0.5b": "500M params, ~1GB VRAM",
            "qwen2.5-1.5b": "1.5B params, ~3GB VRAM",
            "qwen2.5-3b": "3B params, ~6GB VRAM",
            "qwen2.5-7b": "7B params, ~14GB VRAM",
            "qwen2.5-14b": "14B params, ~28GB VRAM",
            "qwen2.5-32b": "32B params, ~64GB VRAM",
            "llama-3.2-1b": "1B params, ~2GB VRAM",
            "llama-3.2-3b": "3B params, ~6GB VRAM",
            "llama-3.1-8b": "8B params, ~16GB VRAM",
            "llama-3.1-70b": "70B params, ~140GB VRAM",
            "mistral-7b": "7B params, ~14GB VRAM",
            "phi-3-mini": "3.8B params, ~8GB VRAM",
            "gemma-2b": "2B params, ~4GB VRAM",
            "gemma-7b": "7B params, ~14GB VRAM",
        }
        info = model_info.get(model, "")
        self.model_info.config(text=info)

    def _on_loss_type_change(self, event=None):
        """Handle loss type change and update dynamic parameters."""
        loss_type = self.loss_type.get()

        # Update description
        descriptions = {
            "grpo": "Standard GRPO - Grouped Relative Policy Optimization",
            "gspo": "GSPO - Generalized SPO with epsilon parameters",
            "dr_grpo": "DR-GRPO - Doubly Robust GRPO variant"
        }
        self.loss_info.config(text=descriptions.get(loss_type, ""))

        # Clear dynamic frame
        for widget in self.dynamic_frame.winfo_children():
            widget.destroy()

        # Create appropriate parameter fields
        if loss_type == "gspo":
            self._create_gspo_params()
        elif loss_type == "dr_grpo":
            self._create_dr_grpo_params()
        else:
            self._create_grpo_params()

    def _create_grpo_params(self):
        """Create standard GRPO parameter fields."""
        # KL Penalty
        ttk.Label(self.dynamic_frame, text="KL Penalty:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.kl_penalty = tk.DoubleVar(value=0.01)
        kl_entry = ttk.Entry(self.dynamic_frame, textvariable=self.kl_penalty, width=15)
        kl_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.dynamic_frame, text="KL divergence coefficient",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=2, padx=5, pady=5)

        # Clip Range
        ttk.Label(self.dynamic_frame, text="Clip Range:").grid(row=0, column=3, sticky=tk.W, padx=(20, 15), pady=5)
        self.clip_range = tk.DoubleVar(value=0.2)
        clip_entry = ttk.Entry(self.dynamic_frame, textvariable=self.clip_range, width=15)
        clip_entry.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.dynamic_frame, text="PPO-style clipping",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=5, padx=5, pady=5)

    def _create_gspo_params(self):
        """Create GSPO-specific parameter fields."""
        # Epsilon
        ttk.Label(self.dynamic_frame, text="Epsilon:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.epsilon = tk.DoubleVar(value=0.2)
        eps_entry = ttk.Entry(self.dynamic_frame, textvariable=self.epsilon, width=15)
        eps_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.dynamic_frame, text="Base epsilon",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=2, padx=5, pady=5)

        # Epsilon High
        ttk.Label(self.dynamic_frame, text="Epsilon High:").grid(row=0, column=3, sticky=tk.W, padx=(20, 15), pady=5)
        self.epsilon_high = tk.DoubleVar(value=0.28)
        eps_high_entry = ttk.Entry(self.dynamic_frame, textvariable=self.epsilon_high, width=15)
        eps_high_entry.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.dynamic_frame, text="One-sided bound",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=0, column=5, padx=5, pady=5)

        # Delta
        ttk.Label(self.dynamic_frame, text="Delta:").grid(row=1, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.delta = tk.DoubleVar(value=1.5)
        delta_entry = ttk.Entry(self.dynamic_frame, textvariable=self.delta, width=15)
        delta_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.dynamic_frame, text="Two-sided bound",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=1, column=2, padx=5, pady=5)

        # Also include standard parameters
        ttk.Label(self.dynamic_frame, text="KL Penalty:").grid(row=1, column=3, sticky=tk.W, padx=(20, 15), pady=5)
        self.kl_penalty = tk.DoubleVar(value=0.01)
        kl_entry = ttk.Entry(self.dynamic_frame, textvariable=self.kl_penalty, width=15)
        kl_entry.grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)

    def _create_dr_grpo_params(self):
        """Create DR-GRPO-specific parameter fields."""
        # KL Penalty
        ttk.Label(self.dynamic_frame, text="KL Penalty:").grid(row=0, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.kl_penalty = tk.DoubleVar(value=0.01)
        kl_entry = ttk.Entry(self.dynamic_frame, textvariable=self.kl_penalty, width=15)
        kl_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        # Clip Range
        ttk.Label(self.dynamic_frame, text="Clip Range:").grid(row=0, column=3, sticky=tk.W, padx=(20, 15), pady=5)
        self.clip_range = tk.DoubleVar(value=0.2)
        clip_entry = ttk.Entry(self.dynamic_frame, textvariable=self.clip_range, width=15)
        clip_entry.grid(row=0, column=4, sticky=tk.W, padx=5, pady=5)

        # Regularization Weight (DR-GRPO specific)
        ttk.Label(self.dynamic_frame, text="Regularization Weight:").grid(row=1, column=0, sticky=tk.W, padx=(5, 15), pady=5)
        self.reg_weight = tk.DoubleVar(value=0.1)
        reg_entry = ttk.Entry(self.dynamic_frame, textvariable=self.reg_weight, width=15)
        reg_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Label(self.dynamic_frame, text="Doubly robust regularization",
                 font=('TkDefaultFont', 9, 'italic')).grid(row=1, column=2, padx=5, pady=5)

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from tab."""
        config = {
            # Model configuration
            "model_name": self.model_name.get(),
            "lora_rank": self.lora_rank.get(),
            "lora_alpha": self.lora_alpha.get(),
            "lora_dropout": self.lora_dropout.get(),
            "target_modules": self.target_modules.get(),

            # Training parameters
            "learning_rate": float(self.learning_rate.get()),
            "batch_size": self.batch_size.get(),
            "gradient_accumulation": self.gradient_accumulation.get(),
            "num_epochs": self.num_epochs.get(),
            "warmup_steps": self.warmup_steps.get(),
            "max_steps": self.max_steps.get(),

            # GRPO Loss configuration
            "loss_type": self.loss_type.get(),
            "mask_truncated_completions": self.mask_truncated.get(),

            # Generation parameters
            "temperature": self.temperature.get(),
            "top_p": self.top_p.get(),
            "top_k": self.top_k.get(),
            "num_generations": self.num_generations.get(),
            "max_new_tokens": self.max_new_tokens.get(),
        }

        # Add loss-specific parameters
        loss_type = self.loss_type.get()
        if loss_type == "gspo":
            config.update({
                "epsilon": self.epsilon.get() if hasattr(self, 'epsilon') else 0.2,
                "epsilon_high": self.epsilon_high.get() if hasattr(self, 'epsilon_high') else 0.28,
                "delta": self.delta.get() if hasattr(self, 'delta') else 1.5,
                "kl_penalty": self.kl_penalty.get() if hasattr(self, 'kl_penalty') else 0.01,
            })
        elif loss_type == "dr_grpo":
            config.update({
                "kl_penalty": self.kl_penalty.get() if hasattr(self, 'kl_penalty') else 0.01,
                "clip_range": self.clip_range.get() if hasattr(self, 'clip_range') else 0.2,
                "reg_weight": self.reg_weight.get() if hasattr(self, 'reg_weight') else 0.1,
            })
        else:  # grpo
            config.update({
                "kl_penalty": self.kl_penalty.get() if hasattr(self, 'kl_penalty') else 0.01,
                "clip_range": self.clip_range.get() if hasattr(self, 'clip_range') else 0.2,
            })

        return config

    def load_config(self, config: Dict[str, Any]):
        """Load configuration into tab."""
        # Model configuration
        if "model_name" in config:
            self.model_name.set(config["model_name"])
            self._update_model_info()
        if "lora_rank" in config:
            self.lora_rank.set(config["lora_rank"])
        if "lora_alpha" in config:
            self.lora_alpha.set(config["lora_alpha"])
        if "lora_dropout" in config:
            self.lora_dropout.set(config.get("lora_dropout", 0.1))
        if "target_modules" in config:
            self.target_modules.set(config.get("target_modules", "q_proj,v_proj,k_proj,o_proj"))

        # Training parameters
        if "learning_rate" in config:
            self.learning_rate.set(str(config["learning_rate"]))
        if "batch_size" in config:
            self.batch_size.set(config["batch_size"])
        if "gradient_accumulation" in config:
            self.gradient_accumulation.set(config.get("gradient_accumulation", 1))
        if "num_epochs" in config:
            self.num_epochs.set(config["num_epochs"])
        if "warmup_steps" in config:
            self.warmup_steps.set(config.get("warmup_steps", 100))
        if "max_steps" in config:
            self.max_steps.set(config.get("max_steps", -1))

        # GRPO Loss configuration
        if "loss_type" in config:
            self.loss_type.set(config.get("loss_type", "grpo"))
            self._on_loss_type_change()
        if "mask_truncated_completions" in config:
            self.mask_truncated.set(config.get("mask_truncated_completions", True))

        # Generation parameters
        if "temperature" in config:
            self.temperature.set(config["temperature"])
        if "top_p" in config:
            self.top_p.set(config["top_p"])
        if "top_k" in config:
            self.top_k.set(config.get("top_k", 50))
        if "num_generations" in config:
            self.num_generations.set(config["num_generations"])
        if "max_new_tokens" in config:
            self.max_new_tokens.set(config.get("max_new_tokens", 512))

        # Load loss-specific parameters based on type
        loss_type = self.loss_type.get()
        if loss_type == "gspo":
            if hasattr(self, 'epsilon') and "epsilon" in config:
                self.epsilon.set(config["epsilon"])
            if hasattr(self, 'epsilon_high') and "epsilon_high" in config:
                self.epsilon_high.set(config["epsilon_high"])
            if hasattr(self, 'delta') and "delta" in config:
                self.delta.set(config["delta"])
        elif loss_type == "dr_grpo":
            if hasattr(self, 'reg_weight') and "reg_weight" in config:
                self.reg_weight.set(config["reg_weight"])

        # Standard GRPO parameters (shared by multiple loss types)
        if hasattr(self, 'kl_penalty') and "kl_penalty" in config:
            self.kl_penalty.set(config["kl_penalty"])
        if hasattr(self, 'clip_range') and "clip_range" in config:
            self.clip_range.set(config["clip_range"])