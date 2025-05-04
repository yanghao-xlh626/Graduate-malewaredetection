import matplotlib.pyplot as plt
import os
import pickle
import time
from IPython.display import display, clear_output




# class Animate:
#     '''
#     控制器:负责图形更新、数据存储和加载
#     anim = Animate(batchsize=32, learn_rate=0.001, max_epochs=30)
#     anim.init_data()          # 数据容器初始化
#     anim.update(...)          # 核心更新方法
#     anim.finalize()           # 保存最终图表并清理中间状态
    
#     '''
#     def __init__(self, batchsize, learn_rate, max_epochs, graph_path='E:\malware\malex\graph'):
#         self.max_epochs = max_epochs
#         self.batchsize = batchsize
#         self.learn_rate = learn_rate
#         self.checkpoint_path = graph_path
#         # self.state_file = os.path.join(graph_path, 'checkpoint.pth')

#         # 确保检查点路径存在
#         os.makedirs(graph_path, exist_ok=True)

#         # 初始化数据
#         self.epochs = []
#         self.train_losses = []
#         self.test_losses = []
#         self.accuracies = []

#         # 初始化图形
#         self.fig, self.ax = plt.subplots(figsize=(10, 6))
#         self.ax.set_title('Training Metrics')
#         self.ax.set_xlabel('Epoch')
#         self.ax.set_ylabel('Loss / Accuracy')
#         self.ax.grid(True)

#         # 画板显示参数
#         self.ax.text(0.25, -0.15, f'Batch Size: {batchsize}',
#                      horizontalalignment='center', verticalalignment='center',
#                      transform=self.ax.transAxes)
#         self.ax.text(0.75, -0.15, f'Learning Rate: {learn_rate}',
#                      horizontalalignment='center', verticalalignment='center',
#                      transform=self.ax.transAxes)

#         # 增加底部空间
#         self.fig.subplots_adjust(bottom=0.2)

#         # 初始化三条线
#         self.line1, = self.ax.plot([], [], 'b-', label='Train Loss')
#         self.line2, = self.ax.plot([], [], 'r-', label='Test Loss')
#         self.line3, = self.ax.plot([], [], 'g-', label='Accuracy')

#         self.ax.legend()

#     def init_data(self):
#         """
#         初始化训练数据容器.
#         """
#         self.epochs = []
#         self.train_losses = []
#         self.test_losses = []
#         self.accuracies = []

#     def update(self, epoch, train_loss, test_loss, accuracy):
#         """
#         更新数据并动态更新画板.

#         参数:
#             epoch (int): 当前训练周期.
#             train_loss (float): 训练损失.
#             test_loss (float): 测试损失.
#             accuracy (float): 准确率.
#         """
#         # self.epochs.append(epoch+1)
#         self.epochs.append(epoch)
#         self.train_losses.append(train_loss)
#         self.test_losses.append(test_loss)
#         self.accuracies.append(accuracy)

#         # 更新数据
#         self.line1.set_data(self.epochs, self.train_losses)
#         self.line2.set_data(self.epochs, self.test_losses)
#         self.line3.set_data(self.epochs, self.accuracies)

#         # 动态更新坐标轴范围
#         self.ax.set_xlim(0, max(self.epochs) + 1)
#         max_value = max(max(self.train_losses), max(self.test_losses), max(self.accuracies))
#         self.ax.set_ylim(0, max_value * 1.1)

#         # 重新绘制图形
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#         # 保存当前图表
#         epoch_file = os.path.join(self.checkpoint_path, f'epoch_{epoch}.png')
#         self.fig.savefig(epoch_file)
#         print(f"已保存 epoch {epoch} 的图表到: {epoch_file}")

#         time.sleep(0.1)  # 确保图形有足够的时间刷新

#     def finalize(self):
#         """
#         在训练结束时保存图表并清理中间状态.
#         """
#         # 确保图表已经正确绘制
#         self.fig.canvas.draw()
#         self.fig.canvas.flush_events()

#         # 保存最终图表
#         final_path = os.path.join(self.checkpoint_path, 'training_metrics.png')
#         self.fig.savefig(final_path)
#         print(f"最终图表已保存到: {final_path}")

#         # 关闭所有图形资源
#         plt.close('all')

#     def save_state_dict(self):
#         """
#         保存 Animate 状态为字典.
#         """
#         return {
#             'epochs': self.epochs,
#             'train_losses': self.train_losses,
#             'test_losses': self.test_losses,
#             'accuracies': self.accuracies
#         }

#     def load_state_dict(self, state):
#         """
#         从字典加载 Animate 状态.
#         """
#         self.epochs = state['epochs']
#         self.train_losses = state['train_losses']
#         self.test_losses = state['test_losses']
#         self.accuracies = state['accuracies']

# if __name__ == "__main__":
#     # 示例用法
#     animate_obj = Animate(batchsize=32, learn_rate=0.001, max_epochs=10)
#     animate_obj.init_data()  # 初始化数据

#     # 模拟训练过程
#     for epoch in range(10):
#         # 模拟训练数据
#         train_loss = 1.0 / (epoch + 1)  # 模拟训练损失逐渐下降
#         test_loss = 0.8 / (epoch + 1)   # 模拟测试损失逐渐下降
#         accuracy = 0.1 * epoch          # 模拟准确率逐渐上升

#         # 更新图表
#         animate_obj.update(epoch, train_loss, test_loss, accuracy)
#         print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}")

#     # 最终保存图表并清理中间状态
#     animate_obj.finalize()

#     # 加载之前保存的状态并继续训练
#     print("\n加载之前保存的状态并继续训练...")
#     for epoch in range(10, 20):
#         # 模拟训练数据
#         train_loss = 1.0 / (epoch + 1)  # 模拟训练损失逐渐下降
#         test_loss = 0.8 / (epoch + 1)   # 模拟测试损失逐渐下降
#         accuracy = 0.1 * epoch          # 模拟准确率逐渐上升

#         # 更新图表
#         animate_obj.update(epoch + 1, train_loss, test_loss, accuracy)
#         print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}")

#     # 最终保存图表并清理中间状态
#     animate_obj.finalize()


import os
import time
import matplotlib.pyplot as plt
import numpy as np

class Animate:
    def __init__(self, batchsize, learn_rate, max_epochs, graph_path='graphs'):
        self.max_epochs = max_epochs
        self.batchsize = batchsize
        self.learn_rate = learn_rate
        self.graph_path = graph_path

        # 确保存储图表的目录存在
        os.makedirs(self.graph_path, exist_ok=True)

        # 初始化存储训练指标的列表
        self.epochs = []
        self.train_losses = []
        self.test_losses = []
        self.accuracies = []

        # 初始化图形和双Y轴
        self.fig, self.ax1 = plt.subplots(figsize=(10, 6))
        self.ax2 = self.ax1.twinx()  # 创建第二个Y轴用于准确率

        # 设置图表标题和轴标签
        self.ax1.set_title('Training Metrics')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss', color='blue')
        self.ax2.set_ylabel('Accuracy', color='green')

        # 设置网格和文本信息
        self.ax1.grid(True)
        self.ax1.text(0.25, -0.15, f'Batch Size: {batchsize}', transform=self.ax1.transAxes, ha='center')
        self.ax1.text(0.75, -0.15, f'Learning Rate: {learn_rate}', transform=self.ax1.transAxes, ha='center')
        self.fig.subplots_adjust(bottom=0.2)

        # 初始化三条线
        self.line1, = self.ax1.plot([], [], 'b-', label='Train Loss')
        self.line2, = self.ax1.plot([], [], 'r-', label='Test Loss')
        self.line3, = self.ax2.plot([], [], 'g-', label='Accuracy')

        # 添加图例
        self.ax1.legend(loc='upper left')
        self.ax2.legend(loc='upper right')

    def update(self, epoch, train_loss, test_loss, accuracy):
        # 将新的训练指标添加到列表中
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.test_losses.append(test_loss)
        self.accuracies.append(accuracy)

        # 更新线条数据
        self.line1.set_data(self.epochs, self.train_losses)
        self.line2.set_data(self.epochs, self.test_losses)
        self.line3.set_data(self.epochs, self.accuracies)

        # 动态更新坐标轴范围
        self.ax1.set_xlim(0, max(self.epochs) + 1)
        self.ax2.set_xlim(0, max(self.epochs) + 1)
        self.ax1.set_ylim(0, max(max(self.train_losses), max(self.test_losses)) * 1.1)
        self.ax2.set_ylim(0, max(self.accuracies) * 1.1)

        # 重新绘制图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 保存当前图表
        epoch_file = os.path.join(self.graph_path, f'epoch_{epoch}.png')
        self.fig.savefig(epoch_file)
        print(f"已保存 epoch {epoch} 的图表到: {epoch_file}")

        time.sleep(0.1)  # 确保图形有足够的时间刷新

    def finalize(self):
        # 保存最终图表
        final_path = os.path.join(self.graph_path, 'training_metrics.png')
        self.fig.savefig(final_path)
        print(f"最终图表已保存到: {final_path}")

        # 关闭图形
        plt.close('all')

    def save_state_dict(self):
        return {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'accuracies': self.accuracies
        }

    def load_state_dict(self, state):
        self.epochs = state.get('epochs', [])
        self.train_losses = state.get('train_losses', [])
        self.test_losses = state.get('test_losses', [])
        self.accuracies = state.get('accuracies', [])

if __name__ == "__main__":
    animate = Animate(batchsize=32, learn_rate=0.001, max_epochs=30, graph_path='graphs')

    # 模拟训练过程
    try:
        for epoch in range(45, 50):  # 模拟训练10个周期
            train_loss = 1.0 / epoch  # 模拟训练损失逐渐下降
            test_loss = 0.8 / epoch   # 模拟测试损失逐渐下降
            accuracy = 0.1 * epoch    # 模拟准确率逐渐上升

            animate.update(epoch, train_loss, test_loss, accuracy)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}")

        animate.finalize()

        # 模拟继续训练
        print("\n继续训练...")
        for epoch in range(11, 16):  # 模拟继续训练5个周期
            train_loss = 1.0 / epoch
            test_loss = 0.8 / epoch
            accuracy = 0.1 * epoch

            animate.update(epoch, train_loss, test_loss, accuracy)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}, Accuracy = {accuracy:.4f}")

        animate.finalize()
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        animate.finalize()