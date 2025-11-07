import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import load_dataset
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from tokenizers.trainers import BpeTrainer
from Transformer import Transformer
import jieba

# 设置中文显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 代理设置
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# 10. 掩码工具函数
def create_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def create_look_ahead_mask(seq_len):
    mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_masks(src, tgt, src_pad_idx, tgt_pad_idx):
    src_mask = create_padding_mask(src, src_pad_idx)
    tgt_pad_mask = create_padding_mask(tgt, tgt_pad_idx)
    tgt_len = tgt.size(1)
    tgt_look_ahead_mask = create_look_ahead_mask(tgt_len).to(device)
    tgt_mask = torch.max(tgt_pad_mask, tgt_look_ahead_mask)
    cross_mask = src_mask
    return src_mask, tgt_mask, cross_mask


# 11. 数据处理
class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenizer, tgt_tokenizer, max_len=128, src_lang='zh', tgt_lang='en'):
        self.data = data
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.processed_data = self._process_data()

    def _process_data(self):
        processed = []
        for item in self.data:
            if isinstance(item, str):
                parts = item.strip().split('\t')
                if len(parts) >= 2:
                    processed.append({'src': parts[0].strip(), 'tgt': parts[1].strip()})
            elif isinstance(item, dict):
                if 'translation' in item:
                    src_text = item['translation'].get(self.src_lang, '').strip()
                    tgt_text = item['translation'].get(self.tgt_lang, '').strip()
                else:
                    src_text = item.get(self.src_lang, '').strip()
                    tgt_text = item.get(self.tgt_lang, '').strip()
                processed.append({'src': src_text, 'tgt': tgt_text})
        return processed

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        src_text = item['src']
        tgt_text = item['tgt']

        # 编码源序列
        src_encoded = self.src_tokenizer.encode(src_text)
        src_ids = src_encoded.ids[:self.max_len]
        src_pad_len = self.max_len - len(src_ids)
        src_ids += [self.src_tokenizer.token_to_id('[PAD]')] * src_pad_len

        # 编码目标序列
        tgt_encoded = self.tgt_tokenizer.encode(tgt_text)
        tgt_ids = tgt_encoded.ids[:self.max_len - 2]  # 预留BOS和EOS
        tgt_ids = [self.tgt_tokenizer.token_to_id('[BOS]')] + tgt_ids + [self.tgt_tokenizer.token_to_id('[EOS]')]
        tgt_pad_len = self.max_len - len(tgt_ids)
        tgt_ids += [self.tgt_tokenizer.token_to_id('[PAD]')] * tgt_pad_len

        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long)
        }

    def __len__(self):
        return len(self.processed_data)


# 12. 训练分词器
def train_tokenizer(dataset, lang, vocab_size=32000, save_path="./tokenizers"):
    os.makedirs(save_path, exist_ok=True)
    tokenizer_path = f"{save_path}/{lang}_tokenizer.json"
    if os.path.exists(tokenizer_path):
        os.remove(tokenizer_path)

    tokenizer = Tokenizer(models.BPE())

    # 中文预处理：jieba分词+空格连接
    def preprocess_chinese(text):
        return " ".join(jieba.cut(text))

    # 预分词器设置
    if lang == "zh":
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    else:
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 分词器其他配置
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]
    )

    # 批量迭代器
    total_texts = 0

    def batch_iterator(batch_size=1000):
        nonlocal total_texts
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            texts = []
            for item in batch:
                if isinstance(item, dict) and 'translation' in item:
                    text = item['translation'].get(lang, '').strip()
                    if text:
                        if lang == "zh":
                            text = preprocess_chinese(text)
                        texts.append(text)
            total_texts += len(texts)
            yield texts

    print(f"开始训练{lang}分词器...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(tokenizer_path)
    print(f"完成训练：共提取{lang}文本 {total_texts} 条")
    return tokenizer


# 13. 训练和评估函数
def train_epoch(model, train_loader, criterion, optimizer, scheduler, src_pad_idx, tgt_pad_idx, clip=1.0):
    model.train()
    total_loss = 0
    batch_losses = []
    for batch in tqdm(train_loader, desc="训练中"):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        tgt_input = tgt[:, :-1]
        tgt_label = tgt[:, 1:]
        src_mask, tgt_mask, cross_mask = create_masks(src, tgt_input, src_pad_idx, tgt_pad_idx)

        optimizer.zero_grad()
        output = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
        loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_label.contiguous().view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        batch_losses.append(loss.item())

    avg_epoch_loss = total_loss / len(train_loader)
    return avg_epoch_loss, batch_losses


def evaluate(model, data_loader, criterion, src_pad_idx, tgt_pad_idx, desc="评估中"):
    model.eval()
    total_loss = 0
    batch_losses = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=desc):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]
            src_mask, tgt_mask, cross_mask = create_masks(src, tgt_input, src_pad_idx, tgt_pad_idx)

            output = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
            loss = criterion(output.contiguous().view(-1, output.size(-1)), tgt_label.contiguous().view(-1))

            total_loss += loss.item()
            batch_losses.append(loss.item())

    avg_epoch_loss = total_loss / len(data_loader)
    return avg_epoch_loss, batch_losses

# 工具函数：按时间命名保存文件
def get_timestamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def save_losses(losses, prefix, timestamp):
    os.makedirs("results/loss_records", exist_ok=True)
    filename = f"results/loss_records/{prefix}_losses_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for i, loss in enumerate(losses):
            f.write(f"batch_{i}: {loss:.6f}\n")
    print(f"已保存{prefix}Batch损失记录到: {filename}")


def plot_epoch_loss_curve(epoch_train_losses, epoch_val_losses, test_loss, timestamp):
    os.makedirs("results/loss_plots", exist_ok=True)
    plt.figure(figsize=(10, 6))

    # 训练集平均损失
    plt.plot(
        range(1, len(epoch_train_losses) + 1),
        epoch_train_losses,
        label="训练集平均损失",
        marker='o',
        linewidth=2
    )

    # 验证集平均损失
    plt.plot(
        range(1, len(epoch_val_losses) + 1),
        epoch_val_losses,
        label="验证集平均损失",
        marker='s',
        linewidth=2,
        color='#ff7f0e'
    )

    # 测试集平均损失
    if test_loss is not None:
        plt.scatter(
            len(epoch_train_losses) + 0.5,
            test_loss,
            label="测试集平均损失",
            color='red',
            s=100,
            zorder=5
        )

    # 图表美化
    plt.xlabel("Epoch序号", fontsize=12)
    plt.ylabel("平均损失值", fontsize=12)
    plt.title("训练集、验证集、测试集平均损失变化（按Epoch）", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.xticks(range(1, len(epoch_train_losses) + 2))

    # 保存图片
    save_path = f"results/loss_plots/epoch_loss_curve_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存Epoch损失曲线到: {save_path}")

# 14. 翻译函数
def translate(model, src_text, src_tokenizer, tgt_tokenizer, max_len=128, src_pad_idx=0, tgt_pad_idx=0, beam_width=5):
    model.eval()
    if not src_text.strip():
        return ""

    # 1. 编码源文本
    src_encoded = src_tokenizer.encode(src_text)
    src_ids = src_encoded.ids[:max_len]
    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = create_padding_mask(src_tensor, src_pad_idx)
    enc_output = model.encoder(src_tensor, src_mask)

    # 2. 初始化束
    bos_id = tgt_tokenizer.token_to_id('[BOS]')
    eos_id = tgt_tokenizer.token_to_id('[EOS]')
    if bos_id is None or eos_id is None:
        raise ValueError("分词器中未找到[BOS]或[EOS]标记")
    beams = [([bos_id], 0.0, False)]

    with torch.no_grad():
        for _ in range(max_len):
            new_beams = []
            for seq_ids, current_score, is_ended in beams:
                if is_ended:
                    new_beams.append((seq_ids, current_score, True))
                    continue
                # 解码当前序列
                tgt_tensor = torch.tensor([seq_ids], dtype=torch.long).to(device)
                tgt_len = tgt_tensor.size(1)
                tgt_mask = create_look_ahead_mask(tgt_len).to(device)
                tgt_pad_mask = create_padding_mask(tgt_tensor, tgt_pad_idx)
                tgt_mask = torch.max(tgt_pad_mask, tgt_mask)
                dec_output = model.decoder(tgt_tensor, enc_output, tgt_mask, src_mask)
                next_token_probs = torch.softmax(model.fc(dec_output[:, -1, :]), dim=-1)
                # 扩展束
                top_k_probs, top_k_ids = next_token_probs.topk(beam_width)
                for prob, token_id in zip(top_k_probs[0], top_k_ids[0]):
                    token_id = token_id.item()
                    new_seq_ids = seq_ids.copy()
                    new_seq_ids.append(token_id)
                    new_score = current_score + (-torch.log(prob)).item()
                    new_ended = (token_id == eos_id)
                    new_beams.append((new_seq_ids, new_score, new_ended))
            # 筛选最优束
            new_beams.sort(key=lambda x: x[1])
            beams = new_beams[:beam_width]
            if all(is_ended for _, _, is_ended in beams):
                break

    # 3. 定义最优序列（关键：确保best_seq在此处定义）
    best_seq = min(beams, key=lambda x: x[1])[0]

    # 4. 调试打印（放在best_seq定义之后！）
    print(f"调试：生成的token ids：{best_seq}")  # 现在能正确引用best_seq
    raw_translated = tgt_tokenizer.decode(best_seq)
    print(f"调试：原始解码结果（含特殊符号）：{raw_translated}")

    # 5. 过滤特殊符号
    translated_text = raw_translated
    for token in ['[BOS]', '[EOS]', '[PAD]', '[UNK]']:
        translated_text = translated_text.replace(token, '').strip()

    return translated_text




# 主函数
def main():
    # 超参数
    batch_size = 16
    max_len = 32
    d_model = 512
    n_layers = 6
    n_heads = 8
    d_ff = 2048
    dropout = 0.2
    num_epochs = 15
    learning_rate = 5e-5

    #生成时间戳
    timestamp = get_timestamp()
    print(f"实验时间戳: {timestamp}")

    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset("iwslt2017", "iwslt2017-zh-en")
    train_data = list(dataset['train'])
    val_data = list(dataset['validation'])
    test_data = list(dataset['test'])
    print(f"训练集大小：{len(train_data)}, 验证集大小：{len(val_data)}, 测试集大小：{len(test_data)}")

    # 训练分词器
    print("训练分词器...")
    src_tokenizer = train_tokenizer(train_data, "zh")
    tgt_tokenizer = train_tokenizer(train_data, "en")

    # 词汇表信息
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    src_pad_idx = src_tokenizer.token_to_id('[PAD]')
    tgt_pad_idx = tgt_tokenizer.token_to_id('[PAD]')
    print(f"源语言词汇表大小: {src_vocab_size}")
    print(f"目标语言词汇表大小: {tgt_vocab_size}")

    # 数据加载器
    print("创建数据加载器...")
    train_dataset = TranslationDataset(train_data, src_tokenizer, tgt_tokenizer, max_len)
    val_dataset = TranslationDataset(val_data, src_tokenizer, tgt_tokenizer, max_len)
    test_dataset = TranslationDataset(test_data, src_tokenizer, tgt_tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    #
    # 初始化模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    ).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=1e-4
    )
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps)

    # 记录损失
    epoch_train_losses = []
    epoch_val_losses = []
    all_train_batch_losses = []
    all_val_batch_losses = []

    # 训练
    print("开始训练...")
    best_val_loss = float('inf')
    model_save_dir = "results/models"
    os.makedirs(model_save_dir, exist_ok=True)
    best_model_path = f"{model_save_dir}/best_transformer_{timestamp}.pth"

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练
        avg_train_loss, train_batch_losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            src_pad_idx, tgt_pad_idx
        )
        epoch_train_losses.append(avg_train_loss)
        all_train_batch_losses.extend(train_batch_losses)

        # 验证
        avg_val_loss, val_batch_losses = evaluate(
            model, val_loader, criterion, src_pad_idx, tgt_pad_idx,
            desc=f"验证Epoch {epoch + 1}"
        )
        epoch_val_losses.append(avg_val_loss)
        all_val_batch_losses.extend(val_batch_losses)

        # 输出Epoch信息
        epoch_time = time.time() - start_time
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"训练平均损失: {avg_train_loss:.4f}")
        print(f"验证平均损失: {avg_val_loss:.4f}")
        print(f"耗时: {epoch_time:.2f}秒")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"保存最佳模型到: {best_model_path}")

    # 测试
    print("\n开始测试...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    avg_test_loss, test_batch_losses = evaluate(
        model, test_loader, criterion, src_pad_idx, tgt_pad_idx,
        desc="测试中"
    )
    print(f"测试平均损失: {avg_test_loss:.4f}")

    # 保存损失记录
    save_losses(all_train_batch_losses, "train_all_batches", timestamp)
    save_losses(all_val_batch_losses, "val_all_batches", timestamp)
    save_losses(test_batch_losses, "test_all_batches", timestamp)

    # 绘制损失曲线
    plot_epoch_loss_curve(epoch_train_losses, epoch_val_losses, avg_test_loss, timestamp)

    #翻译测试
    test_samples = [
        "我喜欢学习自然语言处理。",
        "模型在机器翻译任务中表现很好。",
        "今天天气真好，我们一起去公园吧。"
    ]
    print("\n翻译测试:")
    for sample in test_samples:
        try:
            translation = translate(model, sample, src_tokenizer, tgt_tokenizer, max_len, src_pad_idx, tgt_pad_idx)
            print(f"原文: {sample}")
            print(f"译文: {translation}\n")
        except Exception as e:
            print(f"翻译'{sample}'时出错: {str(e)}\n")


if __name__ == "__main__":

    main()
