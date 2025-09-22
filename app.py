#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV LangChain Agent API Server
一个基于Flask + LangChain + OpenAI的CSV数据查询服务
"""

import os
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentExecutor
import logging

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 全局变量存储agent
agent = None
df = None

def initialize_agent():
    """初始化LangChain Agent和数据"""
    global agent, df

    try:
        # 检查OpenAI API Key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # 读取CSV数据
        csv_path = os.path.join("dataPreProcess", "data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        logger.info(f"✅ Loaded CSV data: {df.shape[0]} rows, {df.shape[1]} columns")

        # 创建OpenAI LLM
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            openai_api_key=openai_api_key
        )

        # 创建Pandas DataFrame Agent with custom prompt
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True,
            allow_dangerous_code=True,  # 允许执行代码
            return_intermediate_steps=False,
            handle_parsing_errors=True,  # 处理解析错误
            prefix="""
You are working with a footfall dataset containing pedestrian count data. The dataset has these key columns:
- Location_Name: The name of the location (e.g., "Park Street")
- Date: The date and time of the measurement
- TotalCount: The main metric - the number of people counted at that location and time
- Hour: The hour of the day (0-23)
- Other columns like LastYear, Previous52DayTimeAvg may contain NaN values - ignore them unless specifically asked

When users ask about footfall, pedestrian count, or "how many people" at a specific location and time, they want the TotalCount value.

Always focus on providing the TotalCount unless the user specifically asks about other metrics.

Examples:
- "footfall at Park Street on 14 Feb 2020 at 11:00" → return the TotalCount for that location/time
- "total people at Market Street" → sum TotalCount for that location
- "busiest hour" → find the hour with highest TotalCount

Be direct and focus on the main metric (TotalCount) that users care about.
"""
        )

        logger.info("✅ LangChain Agent initialized successfully")

    except Exception as e:
        logger.error(f"❌ Failed to initialize agent: {str(e)}")
        raise e

@app.route("/", methods=["GET"])
def health_check():
    """健康检查接口"""
    return "CSV LangChain Agent is running 🚀"

@app.route("/ask", methods=["POST"])
def ask_question():
    """处理用户问题的接口"""
    try:
        # 检查agent是否已初始化
        if agent is None:
            return jsonify({
                "error": "Agent not initialized. Please check server logs.",
                "success": False
            }), 500

        # 获取请求数据
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({
                "error": "Missing 'question' field in request body",
                "success": False
            }), 400

        question = data["question"].strip()
        if not question:
            return jsonify({
                "error": "Question cannot be empty",
                "success": False
            }), 400

        logger.info(f"📝 Received question: {question}")

        # 使用agent处理问题，添加错误处理
        try:
            # 使用invoke方法替代run方法，更加稳定
            result = agent.invoke({"input": question})
            response = result.get("output", str(result))
        except Exception as agent_error:
            # 如果agent遇到解析错误，尝试重新运行
            logger.warning(f"Agent encountered error: {str(agent_error)}")
            if "output parsing error" in str(agent_error).lower():
                # 尝试使用更简单的提示词重新运行
                simplified_question = f"Please answer this question about the data briefly and directly: {question}"
                try:
                    result = agent.invoke({"input": simplified_question})
                    response = result.get("output", str(result))
                except Exception as retry_error:
                    logger.error(f"Retry also failed: {str(retry_error)}")
                    # 如果还是失败，返回一个有用的错误信息
                    response = f"I encountered an error processing your question. Please try rephrasing it more simply. Original question: {question}"
            else:
                raise agent_error

        logger.info(f"✅ Generated response: {response}")

        return jsonify({
            "question": question,
            "answer": response,
            "success": True,
            "data_info": {
                "rows": df.shape[0],
                "columns": df.shape[1],
                "column_names": list(df.columns)
            }
        })

    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/data/info", methods=["GET"])
def data_info():
    """获取数据集信息的接口"""
    if df is None:
        return jsonify({
            "error": "Data not loaded",
            "success": False
        }), 500

    return jsonify({
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "sample_data": df.head(3).to_dict("records"),
        "success": True
    })

@app.errorhandler(404)
def not_found(error):
    """404错误处理"""
    return jsonify({
        "error": "Endpoint not found",
        "success": False
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误处理"""
    return jsonify({
        "error": "Internal server error",
        "success": False
    }), 500

if __name__ == "__main__":
    try:
        # 初始化agent
        initialize_agent()

        # 启动Flask服务
        logger.info("🚀 Starting CSV LangChain Agent API Server...")
        app.run(
            host="0.0.0.0",
            port=5080,
            debug=False,  # 生产环境建议设为False
            threaded=True
        )

    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        print(f"\n❌ Server startup failed: {str(e)}")
        print("Please check:")
        print("1. OPENAI_API_KEY is set in .env file")
        print("2. dataPreProcess/data.csv file exists")
        print("3. All required packages are installed (pip install -r requirements.txt)")
