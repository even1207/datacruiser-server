# CSV LangChain Agent API Server

一个基于 Flask + LangChain + OpenAI 的 CSV 数据查询服务，可以通过自然语言查询 CSV 数据。

## 功能特性

- 🚀 Flask API 服务，监听 `http://0.0.0.0:5000`
- 🤖 集成 LangChain 和 OpenAI GPT-3.5-turbo
- 📊 使用 Pandas DataFrame Agent 处理 CSV 数据查询
- 🔒 环境变量管理 API 密钥
- 📝 支持自然语言查询 CSV 数据

## API 接口

### 1. 健康检查

```
GET /
返回: "CSV LangChain Agent is running 🚀"
```

### 2. 问答接口

```
POST /ask
Content-Type: application/json

请求体:
{
    "question": "你的自然语言问题"
}

响应:
{
    "question": "用户问题",
    "answer": "AI回答",
    "success": true,
    "data_info": {
        "rows": 数据行数,
        "columns": 数据列数,
        "column_names": ["列名列表"]
    }
}
```

### 3. 数据信息接口

```
GET /data/info
返回: 数据集基本信息和样例数据
```

## 安装和使用

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制环境变量模板
cp env_template.txt .env

# 编辑.env文件，添加你的OpenAI API Key
# OPENAI_API_KEY=your_actual_api_key_here
```

### 3. 确保数据文件存在

确保 `dataPreProcess/data.csv` 文件存在并包含有效数据。

### 4. 启动服务

```bash
python app.py
```

服务将在 `http://0.0.0.0:5000` 启动。

### 5. 测试 API

#### 健康检查

```bash
curl http://localhost:5000/
```

#### 数据查询示例

```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "数据集有多少行？"
  }'

curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Park Street location的平均TotalCount是多少？"
  }'

curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "哪个location的TotalCount最高？"
  }'
```

#### 获取数据信息

```bash
curl http://localhost:5000/data/info
```

## 数据格式

当前支持的 CSV 数据格式包含以下字段：

- Location_code: 位置代码
- Location_Name: 位置名称
- Date: 日期时间
- TotalCount: 总计数
- Hour, Day, DayNo, Week: 时间相关字段
- LastWeek, Previous4DayTimeAvg, Previous52DayTimeAvg: 历史数据
- ObjectId, LastYear: 其他标识和历史数据

## 注意事项

1. 需要有效的 OpenAI API Key
2. 确保 CSV 数据文件路径正确
3. 生产环境建议使用 WSGI 服务器（如 Gunicorn）
4. 建议配置适当的日志记录和错误处理

## 依赖包

- Flask: Web 框架
- LangChain: AI 应用框架
- OpenAI: GPT 模型 API
- Pandas: 数据处理
- python-dotenv: 环境变量管理

## 许可证

MIT License
