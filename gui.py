import tkinter as tk
from tkinter import messagebox
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import os
import subprocess
from tkinter import filedialog, Scale  # Add for slider
os.makedirs('summaries', exist_ok=True)

# GUI class
class P2P_GUI:
    def __init__(self, master):
        self.master = master
        master.title("P2P Lending Analysis GUI / P2P 借贷分析 GUI")
        master.geometry("800x700")  # Fixed larger size for full display

        # To enable grid for right-side text box
        main_frame = tk.Frame(master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)

        # Repack existing elements into column 0
        self.load_btn = tk.Button(main_frame, text="Load Cleaned Data / 加载清洗数据", command=self.load_data, font=('Microsoft YaHei', 10))
        self.load_btn.grid(row=0, column=0, pady=10, sticky='w')

        self.viz_btn = tk.Button(main_frame, text="Show Overdue Breakdown Plot / 显示逾期分解图", command=self.show_viz, font=('Microsoft YaHei', 10))
        self.viz_btn.grid(row=1, column=0, pady=10, sticky='w')

        # New button for correlations heatmap (enhancement)
        self.corr_btn = tk.Button(main_frame, text="Show Correlations Heatmap / 显示相关性热图", command=self.show_corr, font=('Microsoft YaHei', 10))
        self.corr_btn.grid(row=2, column=0, pady=10, sticky='w')

        # Add model selection dropdown
        self.model_label = tk.Label(main_frame, text="Select Model / 选择模型:", font=('Microsoft YaHei', 10))
        self.model_label.grid(row=3, column=0, sticky='w')
        self.model_var = tk.StringVar(value='xgboost')
        self.model_dropdown = tk.OptionMenu(main_frame, self.model_var, 'XGBoost (XGBoost)', 'Decision Tree (决策树)', 'Random Forest (随机森林)')
        self.model_dropdown.config(width=20)  # Widen for intuitiveness
        self.model_dropdown.grid(row=4, column=0, pady=5, sticky='w')

        self.val_size_label = tk.Label(main_frame, text="Validation Size (0.1-0.3): / 验证集大小 (0.1-0.3):", font=('Microsoft YaHei', 10))
        self.val_size_label.grid(row=5, column=0, sticky='w')
        self.val_size_slider = Scale(main_frame, from_=0.1, to=0.3, resolution=0.05, orient="horizontal")
        self.val_size_slider.set(0.15)  # Default 0.15
        self.val_size_slider.grid(row=6, column=0, sticky='w')

        self.test_size_label = tk.Label(main_frame, text="Test Size (0.1-0.3): / 测试集大小 (0.1-0.3):", font=('Microsoft YaHei', 10))
        self.test_size_label.grid(row=7, column=0, sticky='w')
        self.test_size_slider = Scale(main_frame, from_=0.1, to=0.3, resolution=0.05, orient="horizontal")
        self.test_size_slider.set(0.15)  # Default 0.15
        self.test_size_slider.grid(row=8, column=0, sticky='w')

        self.train_size_info = tk.Label(main_frame, text="Implied Train Size: 70% / 隐含训练集大小: 70%", font=('Microsoft YaHei', 10))
        self.train_size_info.grid(row=9, column=0, sticky='w')

        # Integrated split bar visualization (canvas for color-coded segments with percentages)
        self.split_bar = tk.Canvas(main_frame, width=300, height=30, bg='white', highlightthickness=1, highlightbackground='gray')
        self.split_bar.grid(row=10, column=0, pady=5, sticky='w')

        # Update train_size label and bar dynamically (nested in init for instance access)
        self.warning_shown = False  # Debounce flag for warnings
        def update_train_size(*args):
            val_size = self.val_size_slider.get()
            test_size = self.test_size_slider.get()
            train_size = 1 - val_size - test_size
            if train_size < 0.4:  # Prevent invalid splits
                if not self.warning_shown:  # Show warning only once per invalid state
                    messagebox.showwarning("Invalid Split", "Train size too small! Adjust sliders.")
                    self.warning_shown = True
                self.split_bar.delete("all")  # Clear bar on invalid
                return
            self.warning_shown = False  # Reset flag on valid split
            self.train_size_info.config(text=f"Implied Train Size: {train_size:.0%} / 隐含训练集大小: {train_size:.0%}")  # Percentage format

            # Update the split bar
            self.split_bar.delete("all")
            bar_width = 300
            train_width = int(bar_width * train_size)
            val_width = int(bar_width * val_size)
            test_width = bar_width - train_width - val_width  # Ensure exact fill

            # Draw segments
            self.split_bar.create_rectangle(0, 0, train_width, 30, fill='green', outline='')
            self.split_bar.create_rectangle(train_width, 0, train_width + val_width, 30, fill='blue', outline='')
            self.split_bar.create_rectangle(train_width + val_width, 0, bar_width, 30, fill='red', outline='')

            # Add percentage text (centered, but only if segment wide enough to avoid overlap)
            min_width_for_text = 40
            if train_width > min_width_for_text:
                self.split_bar.create_text(train_width / 2, 15, text=f"Train: {train_size:.0%}", fill='white', font=('Microsoft YaHei', 8))
            if val_width > min_width_for_text:
                self.split_bar.create_text(train_width + val_width / 2, 15, text=f"Val: {val_size:.0%}", fill='white', font=('Microsoft YaHei', 8))
            if test_width > min_width_for_text:
                self.split_bar.create_text(train_width + val_width + test_width / 2, 15, text=f"Test: {test_size:.0%}", fill='white', font=('Microsoft YaHei', 8))

        # Bind to ButtonRelease-1 for updates only on slider release (prevents popup spam)
        self.val_size_slider.bind("<ButtonRelease-1>", update_train_size)
        self.test_size_slider.bind("<ButtonRelease-1>", update_train_size)

        # Initial draw for defaults
        update_train_size()

        self.train_btn = tk.Button(main_frame, text="Train Selected Model / 训练所选模型", command=self.train_model, font=('Microsoft YaHei', 10))
        self.train_btn.grid(row=11, column=0, pady=10, sticky='w')

        # New button for evaluation
        self.eval_btn = tk.Button(main_frame, text="Run Model Evaluation / 运行模型评估", command=self.run_evaluation, font=('Microsoft YaHei', 10))
        self.eval_btn.grid(row=12, column=0, pady=10, sticky='w')

        # New button for summary
        self.summary_btn = tk.Button(main_frame, text="Generate Summary / 生成摘要", command=self.generate_summary, font=('Microsoft YaHei', 10))
        self.summary_btn.grid(row=13, column=0, pady=10, sticky='w')

        self.open_folder_btn = tk.Button(main_frame, text="Open Summaries Folder / 打开摘要文件夹", command=lambda: os.startfile('summaries'), font=('Microsoft YaHei', 10))
        self.open_folder_btn.grid(row=14, column=0, pady=10, sticky='w')

        self.quit_btn = tk.Button(main_frame, text="Quit / 退出", command=lambda: [self.output_text.insert(tk.END, "退出程序...\n"), master.quit()], font=('Microsoft YaHei', 10))
        self.quit_btn.grid(row=15, column=0, pady=10, sticky='w')

        self.df = None  # To store loaded data

        # Add text box on right (column 1)
        self.output_label = tk.Label(main_frame, text="Output / 输出:", font=('Microsoft YaHei', 10))
        self.output_label.grid(row=0, column=1, sticky='nw')
        self.output_text = tk.Text(main_frame, height=35, width=40, font=('Microsoft YaHei', 10))
        self.output_text.grid(row=1, column=1, rowspan=15, sticky='nsew', padx=10)

        # Add scrollbar
        self.output_scroll = tk.Scrollbar(main_frame, command=self.output_text.yview)
        self.output_scroll.grid(row=1, column=2, rowspan=15, sticky='ns')
        self.output_text.config(yscrollcommand=self.output_scroll.set)

    def load_data(self):
        try:
            self.output_text.insert(tk.END, "开始加载数据...\n")
            lc = pd.read_csv('cleaned_LC.csv', encoding='utf-8')
            self.output_text.insert(tk.END, "已加载 cleaned_LC.csv\n")
            lp = pd.read_csv('../P2P_PPDAI_DATA/LP.csv', encoding='utf-8')
            lp = lp.dropna(subset=['ListingId']).fillna(0)
            self.df = pd.merge(lc, lp, on='ListingId', how='inner')
            self.output_text.insert(tk.END, f"数据合并完成: {self.df.shape} 行/列\n")
            messagebox.showinfo("成功", f"数据加载完成: {self.df.shape} 行/列")
        except Exception as e:
            self.output_text.insert(tk.END, f"加载失败: {str(e)}\n")
            messagebox.showerror("错误", f"加载失败: {str(e)}")

    def show_viz(self):
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        try:
            self.output_text.insert(tk.END, "生成逾期分解图...\n")
            fig, ax = plt.subplots(figsize=(10, 8))
            plot = sns.countplot(x='还款状态', data=self.df, ax=ax, palette='Blues_d')
            ax.set_xlabel('还款状态', fontsize=12)
            ax.set_ylabel('数量', fontsize=12)
            plt.title('逾期分解', fontsize=14)
            ax.grid(True, axis='y', linestyle='--', alpha=0.7)
            for p in plot.patches:
                ax.text(p.get_x() + p.get_width()/2., p.get_height(), f'{int(p.get_height())}', ha='center', va='bottom', fontsize=10)
            note = plt.text(0.5, -0.35, "备注：\n0-‘未还款’\n1-‘已正常还款’\n2-‘已逾期还款’\n3-‘已提前还清该标全部欠款’\n4-‘已部分还款’",
                            ha='center', va='top', transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
            plt.subplots_adjust(bottom=0.4)
            plot_path = os.path.join('summaries', 'overdue_breakdown_gui.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            self.output_text.insert(tk.END, f"图表已保存至 {plot_path}\n")
            messagebox.showinfo("成功", f"图表已保存至 {plot_path}。打开查看。")
            os.startfile(plot_path)
        except Exception as e:
            self.output_text.insert(tk.END, f"绘图失败: {str(e)}\n")
            messagebox.showerror("错误", f"绘图失败: {str(e)}")

    def show_corr(self):
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        try:
            self.output_text.insert(tk.END, "生成相关性热图...\n")
            numeric_df = self.df.select_dtypes(include=['number'])
            corr = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': '相关系数'})
            ax.set_title('相关性热图', fontsize=14)
            plot_path = os.path.join('summaries', 'correlations_gui.png')
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()
            self.output_text.insert(tk.END, f"图表已保存至 {plot_path}\n")
            messagebox.showinfo("成功", f"图表已保存至 {plot_path}。打开查看。")
            os.startfile(plot_path)
        except Exception as e:
            self.output_text.insert(tk.END, f"绘图失败: {str(e)}\n")
            messagebox.showerror("错误", f"绘图失败: {str(e)}")

    def train_model(self):
        if self.df is None:
            messagebox.showwarning("警告", "请先加载数据!")
            return
        try:
            model_type = self.model_var.get()
            self.output_text.delete(1.0, tk.END)  # Clear for new operation
            self.output_text.insert(tk.END, f"开始 {model_type} 模型训练...\n")

            val_size = self.val_size_slider.get()
            test_size = self.test_size_slider.get()
            train_size = 1 - val_size - test_size
            if train_size < 0.4:
                messagebox.showwarning("无效", "训练集太小!")
                return
            self.output_text.insert(tk.END, f"使用比例: 训练 {train_size:.0%} / 验证 {val_size:.0%} / 测试 {test_size:.0%}\n")

            self.output_text.insert(tk.END, "准备数据...\n")
            numeric_df = self.df.select_dtypes(include=['number']).sample(1000)
            if '还款状态' not in numeric_df.columns:
                raise ValueError("未找到目标列 '还款状态'")
            X = numeric_df.drop('还款状态', axis=1, errors='ignore')
            y = numeric_df['还款状态'].apply(lambda x: 1 if x > 0 else 0)
            self.output_text.insert(tk.END, f"数据准备完成: {X.shape[0]} 样本\n")

            self.output_text.insert(tk.END, "分割数据...\n")
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=val_size + test_size, random_state=42, stratify=y)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size / (val_size + test_size), random_state=42, stratify=y_temp)
            self.output_text.insert(tk.END, f"分割完成: 训练 {X_train.shape[0]}, 验证 {X_val.shape[0]}, 测试 {X_test.shape[0]}\n")

            try:
                self.output_text.insert(tk.END, "初始化并训练模型...\n")
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from xgboost import XGBClassifier

                def get_model(mt):
                    mt_lower = mt.lower()
                    if 'xgboost' in mt_lower:
                        return XGBClassifier(eval_metric='logloss', random_state=42)
                    elif 'decision tree' in mt_lower or '决策树' in mt:
                        return DecisionTreeClassifier(class_weight='balanced', random_state=42)
                    elif 'random forest' in mt_lower or '随机森林' in mt:
                        return RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
                    else:
                        raise ValueError(f"未知模型: {mt}")

                model = get_model(model_type)
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False) if model_type == 'xgboost' else model.fit(X_train, y_train)
                self.output_text.insert(tk.END, "模型训练成功\n")
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                self.output_text.insert(tk.END, f"准确率: {acc:.2f}\n")
                messagebox.showinfo("模型结果", f"{model_type} 准确率: {acc:.2f}\n(在子集上训练)")
            except Exception as e:
                import traceback
                error_msg = f"训练失败: {str(e)}\n{traceback.format_exc()}"
                self.output_text.insert(tk.END, error_msg + "\n")
                messagebox.showerror("错误", error_msg)
        except Exception as e:
            messagebox.showerror("错误", f"训练失败: {str(e)}")

    def run_evaluation(self):
        try:
            self.output_text.insert(tk.END, "运行模型评估...\n")
            result = subprocess.run(['python', 'evaluation.py'], capture_output=True, text=True)
            if result.returncode == 0:
                self.output_text.insert(tk.END, result.stdout + "\n")
                messagebox.showinfo("评估结果", "输出已显示在文本框中。")
            else:
                self.output_text.insert(tk.END, f"失败: {result.stderr}\n")
                messagebox.showerror("错误", f"失败: {result.stderr}")
        except Exception as e:
            messagebox.showerror("错误", f"运行评估失败: {str(e)}")

    def generate_summary(self):
        try:
            self.output_text.insert(tk.END, "生成摘要...\n")
            result = subprocess.run(['python', 'summarize.py'], capture_output=True, text=True)
            if result.returncode == 0:
                # 仅输出到文本框，不自动打开文件，也不解析路径
                self.output_text.insert(tk.END, result.stdout + "\n")
                messagebox.showinfo("成功", "摘要已生成（已保存到 summaries/ 文件夹）。")
            else:
                # 不弹出报错窗口，只在输出区域显示
                self.output_text.insert(tk.END, f"生成摘要失败: {result.stderr}\n")
        except Exception as e:
            # 不弹窗，写入输出区域
            self.output_text.insert(tk.END, f"生成摘要失败: {str(e)}\n")

# Run GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = P2P_GUI(root)
    root.mainloop()
