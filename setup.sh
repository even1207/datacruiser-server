#!/bin/bash

# 🚀 CSV LangChain Agent Setup Script
# 自动创建虚拟环境并安装依赖

echo "🚀 Setting up CSV LangChain Agent..."

# 检查Python版本
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

# 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
else
    echo "✅ Virtual environment already exists"
fi

# 激活虚拟环境并安装依赖
echo "⚡ Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install flask python-dotenv pandas langchain langchain-openai langchain-experimental openai pydantic --upgrade

# 检查安装
echo "🔍 Verifying installation..."
python -c "import flask, pandas, langchain_openai, langchain_experimental; print('✅ All packages installed successfully!')"

# 检查.env文件
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
    echo "⚠️  Please edit .env file and add your OpenAI API key"
else
    echo "✅ .env file already exists"
fi

# 检查数据文件
if [ ! -f "dataPreProcess/data.csv" ]; then
    echo "⚠️  Warning: dataPreProcess/data.csv not found"
    echo "   Make sure your CSV data file exists at this path"
else
    echo "✅ Data file found"
fi

echo ""
echo "🎉 Setup complete! Next steps:"
echo "1. Edit .env file and add your OpenAI API key"
echo "2. Make sure dataPreProcess/data.csv exists"
echo "3. Run the server:"
echo "   source venv/bin/activate"
echo "   python app.py"
echo ""
echo "📚 For more information, see README.md"
