# OpenRoute图片分析节点

## 概述

OpenRoute图片分析节点是一套基于OpenRoute API的ComfyUI节点，支持使用多种AI模型进行图片内容分析和理解。这些节点可以集成到ComfyUI工作流中，为图片生成、编辑和分析提供智能支持。

## 功能特性

### 🎯 核心功能
- **图片内容分析**: 使用AI模型分析图片内容，生成详细的文字描述
- **多模型支持**: 支持Google Gemini、OpenAI GPT、Anthropic Claude等多种模型
- **智能图片处理**: 自动调整图片尺寸，优化传输效率
- **批量处理**: 支持批量分析多张图片，提高工作效率

### 🚀 技术特性
- **无缝集成**: 完全兼容ComfyUI工作流系统
- **高效处理**: 智能图片压缩和编码，减少API调用时间
- **错误处理**: 完善的错误处理和日志记录
- **配置灵活**: 支持自定义API密钥、模型选择等参数

## 节点类型

### 1. OpenRoute图片分析 (OpenRouteImageAnalysis)

单张图片分析节点，支持详细的参数配置。

**输入参数:**
- `image`: 要分析的图片 (IMAGE)
- `prompt`: 分析提示词 (STRING)
- `api_key`: OpenRoute API密钥 (STRING)
- `model`: 使用的AI模型 (STRING)
- `max_tokens`: 最大输出token数 (INT)
- `temperature`: 生成温度 (FLOAT)
- `max_width`: 图片最大宽度 (INT)
- `max_height`: 图片最大高度 (INT)
- `quality`: JPEG质量 (INT)

**可选参数:**
- `base_url`: API基础URL (STRING)
- `site_url`: 网站URL (STRING)
- `site_name`: 网站名称 (STRING)
- `output_filename`: 输出文件名 (STRING)

**输出:**
- `analysis_result`: 分析结果文本 (STRING)
- `generated_image`: 生成的图片 (IMAGE)

### 2. OpenRoute批量图片分析 (OpenRouteBatchImageAnalysis)

批量处理多张图片的节点，适合处理大量图片。

**输入参数:**
- `images`: 要分析的图片批次 (IMAGE)
- `prompt`: 分析提示词 (STRING)
- `api_key`: OpenRoute API密钥 (STRING)
- `model`: 使用的AI模型 (STRING)
- `max_tokens`: 最大输出token数 (INT)
- `temperature`: 生成温度 (FLOAT)

**可选参数:**
- `base_url`: API基础URL (STRING)

**输出:**
- `batch_results`: 批量分析结果 (STRING)

## 支持的模型

### Google模型
- `google/gemini-2.5-flash-image-preview` (推荐)
- `google/gemini-2.0-flash-exp`
- `google/gemini-1.5-flash`

### OpenAI模型
- `openai/gpt-4o`
- `openai/gpt-4o-mini`

### Anthropic模型
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-3-haiku`

### 其他模型
- `meta-llama/llama-3.1-8b-instruct`
- `meta-llama/llama-3.1-70b-instruct`
- `mistralai/mistral-7b-instruct`
- `microsoft/phi-3-mini-4k-instruct`

## 安装和配置

### 1. 安装依赖

确保已安装以下Python包：
```bash
pip install openai pillow torch numpy tqdm
```

### 2. 配置API密钥

编辑 `config.py` 文件，设置你的OpenRoute API密钥：
```python
OPENROUTE_CONFIG = {
    "api_key": "your-api-key-here",
    "base_url": "https://openrouter.ai/api/v1",
    # ... 其他配置
}
```

### 3. 重启ComfyUI

配置完成后重启ComfyUI，新节点将出现在节点列表中。

## 使用示例

### 基本图片分析

1. 在ComfyUI中添加 `OpenRoute图片分析` 节点
2. 连接图片输入
3. 设置分析提示词，例如：
   - "请描述这张图片的内容"
   - "分析这张图片的风格和构图"
   - "识别图片中的主要对象和场景"
4. 选择AI模型和参数
5. 运行工作流

### 建筑效果图生成

使用提示词：
```
将这张建筑模型图渲染成写实风格的设计效果图。现代风格建筑，简洁大气，晴天，明亮天空，少量云，少量行人，建筑摄影风格，透视效果丰富细节，高质量，立体感
```

### 批量图片处理

1. 使用 `OpenRoute批量图片分析` 节点
2. 连接图片批次输入
3. 设置统一的提示词
4. 运行工作流，自动处理所有图片

## 工作流示例

### 图片分析工作流
```
图片输入 → OpenRoute图片分析 → 文本输出
                ↓
            图片输出
```

### 批量处理工作流
```
图片批次 → OpenRoute批量图片分析 → 批量结果
```

### 复杂工作流
```
图片输入 → 图片预处理 → OpenRoute图片分析 → 结果分析 → 文本输出
                ↓
            图片输出 → 后处理 → 最终图片
```

## 参数调优建议

### 模型选择
- **高质量分析**: 使用 `google/gemini-2.5-flash-image-preview`
- **快速处理**: 使用 `google/gemini-1.5-flash`
- **创意生成**: 使用 `openai/gpt-4o`

### 温度设置
- **精确分析**: temperature = 0.1-0.3
- **平衡输出**: temperature = 0.5-0.7
- **创意生成**: temperature = 0.8-1.0

### Token限制
- **简短描述**: max_tokens = 200-500
- **详细分析**: max_tokens = 800-1500
- **深度分析**: max_tokens = 2000-4000

## 常见问题

### Q: API调用失败怎么办？
A: 检查网络连接、API密钥有效性，以及API配额是否充足。

### Q: 图片处理速度慢？
A: 可以降低图片质量或尺寸，选择合适的模型，使用批量处理节点。

### Q: 如何获得更好的分析结果？
A: 编写清晰具体的提示词，选择合适的模型，调整温度和token参数。

### Q: 支持哪些图片格式？
A: 支持JPG、PNG、BMP、GIF等常见格式，推荐使用JPG格式。

## 高级用法

### 自定义提示词模板
创建提示词模板文件，根据不同场景使用不同的分析策略。

### 结果后处理
将分析结果连接到其他节点，如文本处理、图片生成等。

### 工作流优化
使用条件节点和循环节点，创建复杂的图片分析工作流。

## 技术支持

如果遇到问题，请检查：
1. 依赖包是否正确安装
2. API密钥是否有效
3. 网络连接是否正常
4. 图片格式和大小是否符合要求

## 更新日志

### v1.0.0
- 初始版本发布
- 支持单张和批量图片分析
- 集成多种AI模型
- 完整的错误处理和日志记录

## 许可证

本项目遵循MIT许可证，详见LICENSE文件。 