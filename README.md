# 🤖 CLasSmart 灵分 pytorch 部分

## 📚 概述 | Overview

CLasSmart 灵分是一款智能垃圾分类系统，用户仅需要提交待分类的图片，即可得到分类结果。本文档介绍了 pytorch 部分的功能和实现，该部分主要负责图像分类模型的训练和推理。

## 🖼️ 工作流程 | Work Flow

pytorch 部分在整个系统中扮演图像分类的核心角色，具体流程如下：

1. **接收图片**：从 Java 后端接收待分类的图片。
2. **模型推理**：使用预训练的 PyTorch 模型对图片进行分类。
3. **返回结果**：将分类结果返回给 Java 后端。

## ⏰ 定时任务 | Scheduled Tasks

pytorch 部分目前不涉及定时任务。模型的训练和更新由 Java 后端触发，通过上传新的数据集进行复训练。

## 🔒 后端安全策略 | Security Policies

pytorch 部分作为后端服务，安全策略主要由 Java 后端管理。pytorch 部分通过 API 与 Java 后端通信，依赖于 Java 后端的安全措施。

## 🌐 分布式扩展性 | Distributed Scalability

pytorch 部分支持分布式部署，可以通过负载均衡器分发推理请求，提升系统吞吐量。模型训练也可以在分布式环境中进行，以加速训练过程。

## 🛠️ 技术栈 | Tech Stack

| 模块 | 技术/组件 |
| --- | --- |
| **pytorch 部分** | PyTorch、FastAPI、Uvicorn |

## 🔮 未来展望 | Future Outlook

pytorch 部分未来将持续优化，计划引入以下技术提升性能与扩展性：

- **模型优化**：使用更先进的模型架构，提升分类准确性。
- **自动化训练**：实现自动化模型训练和部署流程。
- **分布式训练**：利用分布式计算资源加速模型训练。

## 🚀 使用 | Usage

1. 直接运行 `main.py` 文件即可。
2. 命令行运行：

   ```shell
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

## 📂 文件介绍 | File Description

| 文件 | 作用 |
| --- | --- |
| `main.py` | 主文件，用于运行系统 |
| `model.py` | 模型文件，用于定义模型结构 |
| `train.py` | 训练文件，用于训练模型 |
| `utils.py` | 工具文件，用于定义工具函数 |
| `agent.py` | 代理文件，用于调用 LLM API |
| `classdata.json` | 类别数据文件，用于存放类别映射关系 |
| `net.pth` | PyTorch 模型文件，用于加载模型 |
| `data0` | 资源文件夹，用于存放预训练阶段待分类的图片 |
| `.env` | 环境变量文件，用于配置环境变量 |
| `requirements.txt` | 依赖文件，用于安装依赖 |
