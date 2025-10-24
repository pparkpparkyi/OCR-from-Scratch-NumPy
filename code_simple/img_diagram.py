# generate_architecture_diagram.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    # 색상 정의
    color_input = '#E3F2FD'
    color_conv = '#BBDEFB'
    color_rnn = '#90CAF9'
    color_fc = '#64B5F6'
    color_output = '#42A5F5'
    
    y_pos = 19
    
    # 제목
    ax.text(5, y_pos, 'OCR Model Architecture', 
            ha='center', va='top', fontsize=20, fontweight='bold')
    y_pos -= 1.5
    
    # 입력 레이어
    input_box = FancyBboxPatch((2, y_pos-0.8), 6, 0.8, 
                               boxstyle="round,pad=0.1", 
                               facecolor=color_input, 
                               edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(5, y_pos-0.4, 'Input Image\n(1, 32, 32)', 
            ha='center', va='center', fontsize=11, fontweight='bold')
    y_pos -= 1.5
    
    # 화살표 함수
    def add_arrow(y_start, y_end):
        arrow = FancyArrowPatch((5, y_start), (5, y_end),
                               arrowstyle='->', mutation_scale=20, 
                               linewidth=2, color='black')
        ax.add_patch(arrow)
    
    add_arrow(y_pos+0.7, y_pos)
    
    # Conv Block 1
    conv1_box = FancyBboxPatch((1.5, y_pos-1.5), 7, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=color_conv,
                               edgecolor='black', linewidth=2)
    ax.add_patch(conv1_box)
    ax.text(5, y_pos-0.3, 'Conv2D Block 1', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y_pos-0.7, 'Conv2D(1→32, 3×3) + ReLU', 
            ha='center', va='center', fontsize=10)
    ax.text(5, y_pos-1.0, 'MaxPool(2×2)', 
            ha='center', va='center', fontsize=10)
    ax.text(5, y_pos-1.3, 'Output: (32, 16, 16)', 
            ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 2.0
    
    add_arrow(y_pos+0.5, y_pos)
    
    # Conv Block 2
    conv2_box = FancyBboxPatch((1.5, y_pos-1.5), 7, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=color_conv,
                               edgecolor='black', linewidth=2)
    ax.add_patch(conv2_box)
    ax.text(5, y_pos-0.3, 'Conv2D Block 2', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y_pos-0.7, 'Conv2D(32→64, 3×3) + ReLU', 
            ha='center', va='center', fontsize=10)
    ax.text(5, y_pos-1.0, 'MaxPool(2×2)', 
            ha='center', va='center', fontsize=10)
    ax.text(5, y_pos-1.3, 'Output: (64, 8, 8)', 
            ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 2.0
    
    add_arrow(y_pos+0.5, y_pos)
    
    # Reshape
    reshape_box = FancyBboxPatch((2.5, y_pos-0.6), 5, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#FFE0B2',
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(reshape_box)
    ax.text(5, y_pos-0.3, 'Reshape → (1, 4096)', 
            ha='center', va='center', fontsize=10)
    y_pos -= 1.2
    
    add_arrow(y_pos+0.6, y_pos)
    
    # RNN Block
    rnn_box = FancyBboxPatch((1.5, y_pos-2.0), 7, 2.0,
                             boxstyle="round,pad=0.1",
                             facecolor=color_rnn,
                             edgecolor='black', linewidth=2)
    ax.add_patch(rnn_box)
    ax.text(5, y_pos-0.3, 'RNN Layer', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y_pos-0.7, 'SimpleRNN(4096 → 256)', 
            ha='center', va='center', fontsize=10)
    ax.text(5, y_pos-1.1, 'BPTT (Backprop Through Time)', 
            ha='center', va='center', fontsize=9)
    
    # RNN 시각화
    for i, x_offset in enumerate([-1.5, -0.5, 0.5, 1.5]):
        cell = patches.Circle((5 + x_offset, y_pos-1.6), 0.15, 
                             facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(cell)
        if i < 3:
            ax.arrow(5 + x_offset + 0.2, y_pos-1.6, 0.6, 0, 
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    ax.text(5, y_pos-1.85, 'h₀ → h₁ → h₂ → hₜ', 
            ha='center', va='center', fontsize=8, style='italic')
    y_pos -= 2.5
    
    add_arrow(y_pos+0.5, y_pos)
    
    # FC + Softmax
    fc_box = FancyBboxPatch((1.5, y_pos-1.2), 7, 1.2,
                            boxstyle="round,pad=0.1",
                            facecolor=color_fc,
                            edgecolor='black', linewidth=2)
    ax.add_patch(fc_box)
    ax.text(5, y_pos-0.3, 'Classification Head', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(5, y_pos-0.7, 'Linear(256 → 100) + Softmax', 
            ha='center', va='center', fontsize=10)
    ax.text(5, y_pos-1.0, 'Output: Class Probabilities', 
            ha='center', va='center', fontsize=9, style='italic')
    y_pos -= 1.7
    
    add_arrow(y_pos+0.5, y_pos)
    
    # 출력
    output_box = FancyBboxPatch((2, y_pos-0.8), 6, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=color_output,
                                edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(5, y_pos-0.4, 'Prediction: "가" (89.2%)', 
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    
    # 범례
    legend_y = 1.5
    ax.text(1, legend_y, '📊 Model Statistics:', fontsize=10, fontweight='bold')
    ax.text(1, legend_y-0.4, '• Total Parameters: ~500K', fontsize=9)
    ax.text(1, legend_y-0.7, '• Training Time: ~2 hours', fontsize=9)
    ax.text(1, legend_y-1.0, '• Final Accuracy: 89.2%', fontsize=9)
    
    ax.text(6, legend_y, '🔧 Key Techniques:', fontsize=10, fontweight='bold')
    ax.text(6, legend_y-0.4, '• im2col optimization', fontsize=9)
    ax.text(6, legend_y-0.7, '• He initialization', fontsize=9)
    ax.text(6, legend_y-1.0, '• BPTT for RNN', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print("✅ Architecture diagram saved as 'architecture_diagram.png'")
    plt.show()

if __name__ == "__main__":
    create_architecture_diagram()
