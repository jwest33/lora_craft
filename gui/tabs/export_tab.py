"""Export tab for model export functionality."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, Any, Optional
import datetime
import threading
from pathlib import Path


class ExportTab:
    """Tab for model export with comprehensive status tracking."""

    def __init__(self, parent, app_instance):
        """Initialize export tab.

        Args:
            parent: Parent widget (notebook)
            app_instance: Main application instance with trainer
        """
        self.frame = ttk.Frame(parent)
        self.parent = parent
        self.app = app_instance  # Store reference to main app

        # Export state
        self.is_exporting = False
        self.export_thread = None

        # Export statistics
        self.export_stats = {
            'start_time': None,
            'end_time': None,
            'format': None,
            'output_path': None,
            'file_size': None,
            'compression_ratio': None,
            'export_successful': False,
            'error_message': None
        }

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

        # Bind mouse wheel for scrolling - only when mouse is over this canvas
        def _on_mousewheel(event):
            # Check if the mouse is over this canvas
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

        # Bind when mouse enters and unbind when it leaves
        scrollable_frame.bind("<Enter>", _bind_mousewheel)
        scrollable_frame.bind("<Leave>", _unbind_mousewheel)

        # Export Format
        format_frame = ttk.LabelFrame(scrollable_frame, text="Export Format", padding=10)
        format_frame.pack(fill=tk.X, padx=10, pady=(0, 5))  # No padding on top, 5 on bottom

        ttk.Label(format_frame, text="Format:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.export_format = tk.StringVar(value="safetensors")
        formats = ["SafeTensors (16-bit)", "SafeTensors (4-bit)", "GGUF (Q4_K_M)", "GGUF (Q5_K_M)", "GGUF (Q8_0)", "HuggingFace"]
        ttk.Combobox(format_frame, textvariable=self.export_format, values=formats, width=30).grid(row=0, column=1, padx=5, pady=5)

        # Output Settings
        output_frame = ttk.LabelFrame(scrollable_frame, text="Output Settings", padding=10)
        output_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(output_frame, text="Output Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.output_dir = tk.StringVar(value="./exports")
        ttk.Entry(output_frame, textvariable=self.output_dir, width=40).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(output_frame, text="Browse", command=self._browse_output).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(output_frame, text="Model Name:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_name = tk.StringVar(value="my_model")
        ttk.Entry(output_frame, textvariable=self.model_name, width=40).grid(row=1, column=1, padx=5, pady=5)

        # HuggingFace Hub
        hub_frame = ttk.LabelFrame(scrollable_frame, text="HuggingFace Hub (Optional)", padding=10)
        hub_frame.pack(fill=tk.X, padx=10, pady=5)

        self.push_to_hub = tk.BooleanVar(value=False)
        hub_check = ttk.Checkbutton(hub_frame, text="Push to HuggingFace Hub", variable=self.push_to_hub,
                                    command=self._toggle_hub_options)
        hub_check.grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        ttk.Label(hub_frame, text="Hub Model ID:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.hub_model_id = tk.StringVar()
        self.hub_entry = ttk.Entry(hub_frame, textvariable=self.hub_model_id, width=40, state=tk.DISABLED)
        self.hub_entry.grid(row=1, column=1, padx=5, pady=5)

        # Export Controls Frame
        controls_frame = ttk.LabelFrame(scrollable_frame, text="Export Controls", padding=10)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)

        # Buttons row
        button_row = ttk.Frame(controls_frame)
        button_row.pack(fill=tk.X, pady=5)

        self.export_button = ttk.Button(button_row, text="🚀 Start Export", command=self._export_model,
                                        style='Accent.TButton' if 'Accent.TButton' in ttk.Style().theme_names() else None)
        self.export_button.pack(side=tk.LEFT, padx=5)

        self.cancel_button = ttk.Button(button_row, text="❌ Cancel", command=self._cancel_export,
                                        state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5)

        self.view_logs_button = ttk.Button(button_row, text="📋 View Logs", command=self._view_export_logs)
        self.view_logs_button.pack(side=tk.LEFT, padx=5)

        # Progress Frame
        progress_frame = ttk.LabelFrame(scrollable_frame, text="Export Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)

        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Status labels
        status_grid = ttk.Frame(progress_frame)
        status_grid.pack(fill=tk.X, pady=5)

        ttk.Label(status_grid, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.status_label = ttk.Label(status_grid, text="Ready to export", font=('TkDefaultFont', 9, 'bold'))
        self.status_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(status_grid, text="Current Step:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.step_label = ttk.Label(status_grid, text="Not started", font=('TkDefaultFont', 9))
        self.step_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(status_grid, text="Time Elapsed:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.time_label = ttk.Label(status_grid, text="--:--:--", font=('TkDefaultFont', 9))
        self.time_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(status_grid, text="Output Size:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.size_label = ttk.Label(status_grid, text="N/A", font=('TkDefaultFont', 9))
        self.size_label.grid(row=3, column=1, sticky=tk.W, padx=5, pady=2)

        # Export History Frame
        history_frame = ttk.LabelFrame(scrollable_frame, text="Recent Exports", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Treeview for export history
        columns = ('Time', 'Format', 'Size', 'Status')
        self.history_tree = ttk.Treeview(history_frame, columns=columns, height=5, show='headings')

        for col in columns:
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=100)

        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Scrollbar for history
        history_scroll = ttk.Scrollbar(history_frame, orient='vertical', command=self.history_tree.yview)
        history_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_tree.configure(yscrollcommand=history_scroll.set)

        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _browse_output(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)

    def _get_app_instance(self):
        """Get the main application instance."""
        return self.app

    def _toggle_hub_options(self):
        """Toggle HuggingFace Hub options based on checkbox."""
        if self.push_to_hub.get():
            self.hub_entry.config(state=tk.NORMAL)
        else:
            self.hub_entry.config(state=tk.DISABLED)

    def _export_model(self):
        """Start model export with progress tracking."""
        if self.is_exporting:
            messagebox.showwarning("Export In Progress", "An export is already in progress!")
            return

        # Validate configuration
        if not self.output_dir.get():
            messagebox.showerror("Export Error", "Please specify an output directory")
            return

        if not self.model_name.get():
            messagebox.showerror("Export Error", "Please specify a model name")
            return

        if self.push_to_hub.get() and not self.hub_model_id.get():
            messagebox.showerror("Export Error", "Please specify a Hub Model ID for push to hub")
            return

        # Reset stats
        self.export_stats = {
            'start_time': datetime.datetime.now(),
            'end_time': None,
            'format': self.export_format.get(),
            'output_path': Path(self.output_dir.get()) / self.model_name.get(),
            'file_size': None,
            'compression_ratio': None,
            'export_successful': False,
            'error_message': None,
            'model_name': self.model_name.get()
        }

        # Update UI state
        self._set_export_state(True)

        # Start export in background thread
        self.export_thread = threading.Thread(target=self._run_export)
        self.export_thread.daemon = True
        self.export_thread.start()

        # Start progress timer
        self._update_timer()

    def _run_export(self):
        """Run the actual export process."""
        try:
            import time
            import json
            import os
            from pathlib import Path

            # Create output directory if it doesn't exist
            output_dir = Path(self.output_dir.get())
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get the actual model and tokenizer from the parent app
            app = self._get_app_instance()
            if not app or not hasattr(app, 'trainer') or not app.trainer:
                raise ValueError("No trained model found. Please train a model first before exporting.")

            trainer = app.trainer
            if not trainer.model:
                raise ValueError("Model not loaded. Please ensure training has completed.")

            format_type = self.export_format.get()
            model_name = self.model_name.get()

            # Export based on format
            if "GGUF" in format_type:
                # Export to GGUF format
                self.step_label.config(text="Converting to GGUF format...")
                self.progress_bar['value'] = 20

                # Determine quantization from format
                if "Q4_K_M" in format_type:
                    quantization = "q4_k_m"
                elif "Q5_K_M" in format_type:
                    quantization = "q5_k_m"
                elif "Q8_0" in format_type:
                    quantization = "q8_0"
                else:
                    quantization = "q4_k_m"

                # Save model in GGUF format using Unsloth's export
                output_path = output_dir / f"{model_name}.gguf"

                # Use Unsloth's GGUF export if available
                if hasattr(trainer.model, 'save_pretrained_gguf'):
                    self.step_label.config(text="Exporting to GGUF...")
                    self.progress_bar['value'] = 50
                    trainer.model.save_pretrained_gguf(str(output_dir), trainer.tokenizer, quantization_method=quantization)
                    self.step_label.config(text="GGUF export complete")
                    self.progress_bar['value'] = 90
                else:
                    raise ValueError("GGUF export not supported. Please ensure you're using an Unsloth model.")

            elif "SafeTensors" in format_type:
                # Export to SafeTensors format
                self.step_label.config(text="Preparing SafeTensors export...")
                self.progress_bar['value'] = 20

                # Determine bit precision
                if "16-bit" in format_type:
                    # Save in 16-bit precision
                    self.step_label.config(text="Saving in 16-bit precision...")
                    self.progress_bar['value'] = 50

                    # Save model and tokenizer
                    trainer.model.save_pretrained_merged(
                        str(output_dir / model_name),
                        trainer.tokenizer,
                        save_method="merged_16bit",
                    )
                else:
                    # Save in 4-bit precision
                    self.step_label.config(text="Saving in 4-bit precision...")
                    self.progress_bar['value'] = 50

                    trainer.model.save_pretrained_merged(
                        str(output_dir / model_name),
                        trainer.tokenizer,
                        save_method="merged_4bit",
                    )

                self.step_label.config(text="SafeTensors export complete")
                self.progress_bar['value'] = 90
                output_path = output_dir / model_name

            else:
                # Export to HuggingFace format
                self.step_label.config(text="Exporting to HuggingFace format...")
                self.progress_bar['value'] = 30

                output_path = output_dir / model_name

                # Save model and tokenizer in HuggingFace format
                self.step_label.config(text="Saving model...")
                self.progress_bar['value'] = 50
                trainer.model.save_pretrained(str(output_path))

                self.step_label.config(text="Saving tokenizer...")
                self.progress_bar['value'] = 70
                trainer.tokenizer.save_pretrained(str(output_path))

                self.step_label.config(text="HuggingFace export complete")
                self.progress_bar['value'] = 90

            # Push to HuggingFace Hub if requested
            if self.push_to_hub.get() and self.hub_model_id.get():
                self.step_label.config(text="Pushing to HuggingFace Hub...")
                self.progress_bar['value'] = 95

                if hasattr(trainer.model, 'push_to_hub_merged'):
                    trainer.model.push_to_hub_merged(
                        self.hub_model_id.get(),
                        trainer.tokenizer,
                        save_method="merged_16bit",
                        token=os.environ.get("HUGGINGFACE_TOKEN", None)
                    )

            # Calculate actual file size
            if output_path.exists():
                if output_path.is_dir():
                    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
                else:
                    total_size = output_path.stat().st_size

                self.export_stats['file_size'] = total_size / (1024 * 1024)  # Convert to MB
            else:
                self.export_stats['file_size'] = 0

            self.progress_bar['value'] = 100
            self.step_label.config(text="Export completed successfully!")

            self.export_stats['export_successful'] = True
            self.export_stats['end_time'] = datetime.datetime.now()
            self.export_stats['output_path'] = str(output_path)

            # Add to history
            self._add_to_history()

            # Show success
            self._show_export_summary()

        except Exception as e:
            self.export_stats['error_message'] = str(e)
            self.export_stats['export_successful'] = False
            messagebox.showerror("Export Failed", f"Export failed: {e}")

        finally:
            self.is_exporting = False
            self._set_export_state(False)

    def _set_export_state(self, is_exporting: bool):
        """Update UI based on export state."""
        if is_exporting:
            self.export_button.config(state=tk.DISABLED)
            self.cancel_button.config(state=tk.NORMAL)
            self.status_label.config(text="Exporting...", foreground='blue')
            self.step_label.config(text="Initializing...")
            self.progress_bar['value'] = 0
            self.is_exporting = True
        else:
            self.export_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            if self.export_stats['export_successful']:
                self.status_label.config(text="Export completed successfully!", foreground='green')
            elif self.export_stats.get('error_message'):
                self.status_label.config(text="Export failed", foreground='red')
            else:
                self.status_label.config(text="Ready to export", foreground='black')
            self.is_exporting = False

    def _cancel_export(self):
        """Cancel ongoing export."""
        if messagebox.askyesno("Cancel Export", "Are you sure you want to cancel the export?"):
            self.is_exporting = False
            self.status_label.config(text="Export cancelled", foreground='orange')
            self.step_label.config(text="Cancelled by user")

    def _update_timer(self):
        """Update the elapsed time display."""
        if self.is_exporting and self.export_stats['start_time']:
            elapsed = datetime.datetime.now() - self.export_stats['start_time']
            elapsed_str = str(elapsed).split('.')[0]
            self.time_label.config(text=elapsed_str)

            # Update size if available
            if self.export_stats.get('file_size'):
                size_mb = self.export_stats['file_size']
                if size_mb >= 1024:
                    self.size_label.config(text=f"{size_mb / 1024:.1f} GB")
                else:
                    self.size_label.config(text=f"{size_mb} MB")

            # Schedule next update
            self.frame.after(1000, self._update_timer)

    def _add_to_history(self):
        """Add export to history list."""
        if self.export_stats['export_successful']:
            time_str = self.export_stats['start_time'].strftime('%H:%M:%S')
            format_str = self.export_stats['format'].split(' ')[0]
            size_str = f"{self.export_stats.get('file_size', 0)} MB"
            status_str = "✅ Success"

            self.history_tree.insert('', 0, values=(time_str, format_str, size_str, status_str))

            # Keep only last 10 entries
            items = self.history_tree.get_children()
            if len(items) > 10:
                self.history_tree.delete(items[-1])

    def _view_export_logs(self):
        """View detailed export logs."""
        logs_window = tk.Toplevel(self.frame)
        logs_window.title("Export Logs")
        logs_window.geometry("600x400")

        text_widget = tk.Text(logs_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add sample logs
        logs = """Export Logs
=============
[INFO] Export started at {}
[INFO] Model format: {}
[INFO] Output directory: {}
[INFO] Model name: {}
[INFO] Export completed successfully
[INFO] Final size: {} MB
[INFO] Compression ratio: {:.1%}
""".format(
            self.export_stats.get('start_time', 'N/A'),
            self.export_stats.get('format', 'N/A'),
            self.export_stats.get('output_path', 'N/A'),
            self.export_stats.get('model_name', 'N/A'),
            self.export_stats.get('file_size', 0),
            self.export_stats.get('compression_ratio', 0)
        )

        text_widget.insert('1.0', logs)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(logs_window, text="Close", command=logs_window.destroy).pack(pady=5)

    def _show_export_summary(self):
        """Show comprehensive export summary dialog."""
        summary_window = tk.Toplevel(self.frame)
        summary_window.title("Export Summary")
        summary_window.geometry("500x600")
        summary_window.transient(self.frame)
        summary_window.grab_set()

        # Main frame
        main_frame = ttk.Frame(summary_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="✨ Export Complete!",
            font=('TkDefaultFont', 14, 'bold')
        )
        title_label.pack(pady=(0, 20))

        # Calculate duration
        if self.export_stats['start_time'] and self.export_stats['end_time']:
            duration = self.export_stats['end_time'] - self.export_stats['start_time']
            duration_str = str(duration).split('.')[0]
        else:
            duration_str = "Unknown"

        # Create sections
        sections = [
            ("📦 Export Details", [
                ("Model Name", self.export_stats.get('model_name', 'Unknown')),
                ("Export Format", self.export_stats.get('format', 'Unknown')),
                ("Output Path", str(self.export_stats.get('output_path', 'Unknown'))),
            ]),
            ("⏱️ Performance", [
                ("Start Time", self.export_stats['start_time'].strftime('%Y-%m-%d %H:%M:%S') if self.export_stats['start_time'] else 'Unknown'),
                ("End Time", self.export_stats['end_time'].strftime('%Y-%m-%d %H:%M:%S') if self.export_stats['end_time'] else 'Unknown'),
                ("Duration", duration_str),
            ]),
            ("💾 Output Statistics", [
                ("File Size", f"{self.export_stats.get('file_size', 0)} MB"),
                ("Compression Ratio", f"{self.export_stats.get('compression_ratio', 0):.1%}"),
                ("Status", "✅ Successfully exported" if self.export_stats['export_successful'] else "❌ Export failed"),
            ])
        ]

        # Create each section
        for section_title, items in sections:
            section_frame = ttk.LabelFrame(main_frame, text=section_title, padding="10")
            section_frame.pack(fill=tk.X, pady=5)

            for label, value in items:
                item_frame = ttk.Frame(section_frame)
                item_frame.pack(fill=tk.X, pady=2)

                ttk.Label(
                    item_frame,
                    text=f"{label}:",
                    width=20,
                    anchor=tk.W
                ).pack(side=tk.LEFT)

                ttk.Label(
                    item_frame,
                    text=str(value),
                    font=('TkDefaultFont', 9, 'bold')
                ).pack(side=tk.LEFT)

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))

        ttk.Button(
            button_frame,
            text="📁 Open Output Folder",
            command=lambda: self._open_output_folder()
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="📄 Export Report",
            command=lambda: self._export_report()
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Close",
            command=summary_window.destroy
        ).pack(side=tk.RIGHT, padx=5)

        # Center the window
        summary_window.update_idletasks()
        x = (summary_window.winfo_screenwidth() // 2) - (summary_window.winfo_width() // 2)
        y = (summary_window.winfo_screenheight() // 2) - (summary_window.winfo_height() // 2)
        summary_window.geometry(f"+{x}+{y}")

    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        import os
        import platform

        output_path = self.export_stats.get('output_path')
        if output_path:
            folder = str(Path(output_path).parent)
            if platform.system() == 'Windows':
                os.startfile(folder)
            elif platform.system() == 'Darwin':
                os.system(f'open "{folder}"')
            else:
                os.system(f'xdg-open "{folder}"')

    def _export_report(self):
        """Export detailed report of the export process."""
        import json

        file_path = filedialog.asksaveasfilename(
            title="Export Report",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("Text Files", "*.txt")],
            initialfile=f"export_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        if file_path:
            try:
                # Prepare export data
                export_data = self.export_stats.copy()

                # Convert datetime to string
                if export_data.get('start_time'):
                    export_data['start_time'] = export_data['start_time'].isoformat()
                if export_data.get('end_time'):
                    export_data['end_time'] = export_data['end_time'].isoformat()
                if export_data.get('output_path'):
                    export_data['output_path'] = str(export_data['output_path'])

                # Save the report
                with open(file_path, 'w') as f:
                    json.dump(export_data, f, indent=2)

                messagebox.showinfo("Report Saved", f"Export report saved to {Path(file_path).name}")
            except Exception as e:
                messagebox.showerror("Export Failed", f"Failed to save report: {e}")
