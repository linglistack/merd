# 快速启动和测试
# 相比完整的 main.py，demo_main.py 有以下优势：
# 启动速度快 - 不需要加载完整的 Meridian 库和 TensorFlow
# 依赖少 - 只需要基本的 FastAPI 依赖
# 资源占用低 - 不需要 GPU 或大量内存
# 调试友好 - 代码简单，容易理解和修改

# 简化的 API 接口
# 提供了与完整版本相同的 API 端点，但返回模拟数据：
# /analyze/roi-analysis - ROI 分析
# /analyze/response-curves - 响应曲线分析
# /analyze/predictive-accuracy - 预测准确性
# /analyze/optimal-frequency - 最优频次分析

# 适合用于：
# 向客户或团队演示功能
# 培训新团队成员
# 验证 API 设计

# 开发阶段：使用 demo_main.py 进行快速迭代
# 生产环境：使用完整的 main.py 进行真实分析


from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from typing import List, Optional
import io
import csv

app = FastAPI(title="Meridian API Demo", description="演示版本的 Meridian API")

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    data: dict
    confidence_level: float = 0.9
    selected_geos: Optional[List[str]] = None
    selected_times: Optional[List[str]] = None

@app.get("/")
async def root():
    return {
        "message": "欢迎使用 Meridian API 演示版本",
        "version": "demo",
        "description": "这是一个简化的演示版本，展示了 Meridian 营销组合建模的基本功能"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "API 正在运行"}

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    """上传和预览数据文件"""
    if not file.filename.endswith(('.csv', '.xlsx')):
        raise HTTPException(status_code=400, detail="只支持 CSV 和 Excel 文件")

    try:
        # 读取文件内容
        contents = await file.read()

        if file.filename.endswith('.csv'):
            # 读取 CSV 文件
            csv_data = contents.decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(csv_data))
            data = list(csv_reader)
        else:
            # 对于 Excel 文件，返回一个演示消息
            return {
                "message": "Excel 文件上传成功",
                "filename": file.filename,
                "note": "在完整版本中，这里会解析 Excel 文件"
            }

        # 返回前几行数据作为预览
        preview = data[:5] if len(data) > 5 else data

        return {
            "message": "数据上传成功",
            "filename": file.filename,
            "rows": len(data),
            "columns": list(data[0].keys()) if data else [],
            "preview": preview
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件处理错误: {str(e)}")

@app.get("/sample-data")
async def get_sample_data():
    """获取示例数据"""
    sample_data = {
        "data": [
            {"geo": "California", "date": "2023-01-01", "kpi": 1000, "population": 40000000, "tv_spend": 500, "digital_spend": 300},
            {"geo": "California", "date": "2023-01-02", "kpi": 1050, "population": 40000000, "tv_spend": 520, "digital_spend": 310},
            {"geo": "Texas", "date": "2023-01-01", "kpi": 800, "population": 30000000, "tv_spend": 400, "digital_spend": 250},
            {"geo": "Texas", "date": "2023-01-02", "kpi": 820, "population": 30000000, "tv_spend": 410, "digital_spend": 260},
            {"geo": "New York", "date": "2023-01-01", "kpi": 900, "population": 20000000, "tv_spend": 450, "digital_spend": 280},
            {"geo": "New York", "date": "2023-01-02", "kpi": 920, "population": 20000000, "tv_spend": 460, "digital_spend": 290}
        ],
        "columns": ["geo", "date", "kpi", "population", "tv_spend", "digital_spend"]
    }

    return {
        "message": "示例营销数据",
        "description": "这是一个包含地理位置、日期、KPI、人口和媒体支出的示例数据集",
        "sample_data": sample_data
    }

@app.post("/analyze/roi-analysis")
async def analyze_roi(request: AnalysisRequest):
    """ROI 分析演示"""
    # 这里是演示版本，返回模拟的分析结果
    return {
        "message": "ROI 分析完成",
        "note": "这是演示版本的结果",
        "analysis": {
            "tv_roi": 2.5,
            "digital_roi": 3.2,
            "total_roi": 2.8,
            "confidence_level": request.confidence_level,
            "analysis_summary": "数字广告的 ROI 高于电视广告，整体投资回报率为 2.8"
        }
    }

@app.post("/analyze/response-curves")
async def analyze_response_curves(request: AnalysisRequest):
    """响应曲线分析演示"""
    return {
        "message": "响应曲线分析完成",
        "note": "这是演示版本的结果",
        "analysis": {
            "tv_curve": {
                "saturation_point": 1000,
                "max_response": 2000,
                "efficiency": 0.75
            },
            "digital_curve": {
                "saturation_point": 800,
                "max_response": 1800,
                "efficiency": 0.85
            },
            "summary": "数字广告达到饱和点更快，但效率更高"
        }
    }

@app.post("/analyze/predictive-accuracy")
async def analyze_predictive_accuracy(request: AnalysisRequest):
    """预测准确性分析演示"""
    return {
        "message": "预测准确性分析完成",
        "note": "这是演示版本的结果",
        "analysis": {
            "mape": 12.5,  # 平均绝对百分比误差
            "rmse": 0.08,  # 均方根误差
            "r_squared": 0.85,  # 决定系数
            "accuracy_grade": "良好",
            "summary": "模型预测准确性良好，R² 值为 0.85"
        }
    }

@app.post("/analyze/optimal-frequency")
async def analyze_optimal_frequency(request: AnalysisRequest):
    """最优频次分析演示"""
    return {
        "message": "最优频次分析完成",
        "note": "这是演示版本的结果",
        "analysis": {
            "tv_optimal_frequency": 4.2,
            "digital_optimal_frequency": 6.8,
            "frequency_efficiency": {
                "tv": "每周 4-5 次接触效果最佳",
                "digital": "每周 6-7 次接触效果最佳"
            },
            "summary": "数字广告的最优接触频次高于电视广告"
        }
    }

@app.get("/features")
async def get_features():
    """获取功能列表"""
    return {
        "available_features": [
            {
                "name": "ROI 分析",
                "endpoint": "/analyze/roi-analysis",
                "description": "分析各媒体渠道的投资回报率"
            },
            {
                "name": "响应曲线分析",
                "endpoint": "/analyze/response-curves",
                "description": "分析媒体投入与效果的关系曲线"
            },
            {
                "name": "预测准确性",
                "endpoint": "/analyze/predictive-accuracy",
                "description": "评估模型的预测准确性"
            },
            {
                "name": "最优频次分析",
                "endpoint": "/analyze/optimal-frequency",
                "description": "分析广告投放的最佳频次"
            }
        ],
        "note": "这是演示版本，完整版本包含更多高级功能，如 Hill 曲线、Adstock 衰减等"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
