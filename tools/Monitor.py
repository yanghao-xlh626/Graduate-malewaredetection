class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.count = 0

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.delta:
            self.count += 1
            if self.count >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.count = 0





# early_stopping = EarlyStopping(patience=5, delta=0.001)
# for epoch in range(max_epochs):
#     # 训练代码
#     model.train()
#     # ...
    
#     # 验证代码
#     model.eval()
#     val_loss = # 计算验证损
    
#     # 检查早停条件
#     early_stopping(val_loss, model)
#     if early_stopping.early_stop:
#         print("Early stopping triggered")
#         break