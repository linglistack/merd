import sys
import os
from pathlib import Path

# Add the meridian package directory to Python path
MERIDIAN_DIR = Path(__file__).parent / "meridian"
sys.path.append(str(MERIDIAN_DIR))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from typing import List, Optional
from io import BytesIO
import xarray as xr
import logging

# Import meridian package
from meridian import model, analysis
from meridian.data.input_data import InputData

app = FastAPI(title="Meridian API", description="API for the Meridian package")

# Get allowed origins from environment variable or use default
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5000,https://your-frontend-url.vercel.app"
).split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Export handler for Vercel
handler = app

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisRequest(BaseModel):
    data: dict
    confidence_level: float = 0.9
    selected_geos: Optional[List[str]] = None
    selected_times: Optional[List[str]] = None
    # Sampling parameters
    n_draws: Optional[int] = 1000  # Default number of draws for prior sampling
    n_chains: Optional[int] = 4    # Default number of MCMC chains
    n_adapt: Optional[int] = 1000  # Default number of adaptation steps
    n_burnin: Optional[int] = 1000 # Default number of burn-in steps
    n_keep: Optional[int] = 1000   # Default number of samples to keep

@app.get("/")
async def root():
    return {"message": "Welcome to Meridian API"}

def create_input_data(data_dict):
    """Create InputData object from dictionary data."""
    try:
        logger.info("Starting InputData creation process")

        # Convert the dictionary back to a DataFrame
        df = pd.DataFrame(data_dict["data"], columns=data_dict["columns"])
        logger.info(f"DataFrame created with shape: {df.shape}")

        # Validate required columns
        required_columns = ['geo', 'date', 'kpi', 'population']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}. Required: {', '.join(required_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log column information
        logger.info(f"Available columns: {', '.join(df.columns)}")

        # Ensure proper data types
        df['date'] = pd.to_datetime(df['date'])
        df['kpi'] = pd.to_numeric(df['kpi'], errors='coerce')
        df['population'] = pd.to_numeric(df['population'], errors='coerce')

        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            null_cols = [f"{col} ({count} nulls)" for col, count in null_counts.items() if count > 0]
            raise ValueError(f"Found null values in columns: {', '.join(null_cols)}")

        # Sort data
        df = df.sort_values(['geo', 'date'])

        # Convert dates to datetime objects and then to YYYY-MM-DD format
        df['date'] = pd.to_datetime(df['date'])
        unique_dates = df['date'].dt.strftime('%Y-%m-%d').unique()
        unique_geos = df['geo'].unique()

        # Create KPI array
        kpi_pivot = df.pivot(index='geo', columns='date', values='kpi')
        kpi = xr.DataArray(
            kpi_pivot.values,
            dims=['geo', 'time'],
            coords={
                'geo': unique_geos,
                'time': unique_dates  # Now using formatted dates
            },
            name='kpi'
        )
        logger.info(f"Created KPI array with shape: {kpi.shape}")

        # Create population array
        population_data = df.groupby('geo')['population'].first()
        population = xr.DataArray(
            population_data.values,
            dims=['geo'],
            coords={'geo': unique_geos},
            name='population'
        )
        logger.info(f"Created population array with shape: {population.shape}")

        # Initialize dictionaries to store media data
        media_data = {}
        media_spend_data = {}

        # Process each media channel
        for spend_col in [col for col in df.columns if col.endswith('_spend')]:
            channel_name = spend_col.replace('_spend', '')

            # Create media spend array
            spend_pivot = df.pivot(index='geo', columns='date', values=spend_col)
            media_spend_data[channel_name] = xr.DataArray(
                spend_pivot.values,
                dims=['geo', 'time'],
                coords={
                    'geo': unique_geos,
                    'time': unique_dates  # Using formatted dates
                },
                name=f"{channel_name}_spend"
            )
            logger.info(f"Created media spend array for {channel_name}")

            # For media execution, we'll use the spend values initially
            media_data[channel_name] = xr.DataArray(
                spend_pivot.values,
                dims=['geo', 'media_time'],
                coords={
                    'geo': unique_geos,
                    'media_time': unique_dates  # Using formatted dates
                },
                name=channel_name
            )
            logger.info(f"Created media array for {channel_name}")

        # Combine all media channels into a single DataArray
        media_channels = list(media_data.keys())
        media_values = np.stack([media_data[ch].values for ch in media_channels], axis=-1)
        media = xr.DataArray(
            media_values,
            dims=['geo', 'media_time', 'media_channel'],
            coords={
                'geo': unique_geos,
                'media_time': unique_dates,  # Using formatted dates
                'media_channel': media_channels
            },
            name='media'
        )

        # Do the same for media spend
        media_spend_values = np.stack([media_spend_data[ch].values for ch in media_channels], axis=-1)
        media_spend = xr.DataArray(
            media_spend_values,
            dims=['geo', 'time', 'media_channel'],
            coords={
                'geo': unique_geos,
                'time': unique_dates,  # Using formatted dates
                'media_channel': media_channels
            },
            name='media_spend'
        )

        logger.info(f"Created media array with shape: {media.shape} and dimensions: {media.dims}")
        logger.info(f"Created media_spend array with shape: {media_spend.shape} and dimensions: {media_spend.dims}")

        # Validate dimensions before creating InputData object
        if 'media' in locals() and 'media_spend' in locals():
            # Check media dimensions
            if media.dims != ('geo', 'media_time', 'media_channel'):
                raise ValueError(f"Media array has incorrect dimensions: {media.dims}. Expected: ('geo', 'media_time', 'media_channel')")

            # Check media_spend dimensions
            if media_spend.dims != ('geo', 'time', 'media_channel'):
                raise ValueError(f"Media spend array has incorrect dimensions: {media_spend.dims}. Expected: ('geo', 'time', 'media_channel')")

            # Check KPI dimensions
            if kpi.dims != ('geo', 'time'):
                raise ValueError(f"KPI array has incorrect dimensions: {kpi.dims}. Expected: ('geo', 'time')")

            # Check population dimensions
            if population.dims != ('geo',):
                raise ValueError(f"Population array has incorrect dimensions: {population.dims}. Expected: ('geo',)")

            # Check that dimensions align
            if not (kpi.shape[0] == media.shape[0] and kpi.shape[1] == media.shape[1]):
                raise ValueError(f"Dimension mismatch: KPI shape {kpi.shape} doesn't match media shape {media.shape}")

        # Create InputData object
        try:
            logger.info("Creating InputData object with media data")

            # Validate dimensions before creating InputData object
            if 'media' in locals() and 'media_spend' in locals():
                # Check media dimensions
                if media.dims != ('geo', 'media_time', 'media_channel'):
                    raise ValueError(f"Media array has incorrect dimensions: {media.dims}. Expected: ('geo', 'media_time', 'media_channel')")

                # Check media_spend dimensions
                if media_spend.dims != ('geo', 'time', 'media_channel'):
                    raise ValueError(f"Media spend array has incorrect dimensions: {media_spend.dims}. Expected: ('geo', 'time', 'media_channel')")

                # Check KPI dimensions
                if kpi.dims != ('geo', 'time'):
                    raise ValueError(f"KPI array has incorrect dimensions: {kpi.dims}. Expected: ('geo', 'time')")

                # Check population dimensions
                if population.dims != ('geo',):
                    raise ValueError(f"Population array has incorrect dimensions: {population.dims}. Expected: ('geo',)")

                # Check that dimensions align
                if not (kpi.shape[0] == media.shape[0] and kpi.shape[1] == media.shape[1]):
                    raise ValueError(f"Dimension mismatch: KPI shape {kpi.shape} doesn't match media shape {media.shape}")

            # Create InputData object
            input_data_obj = InputData(
                kpi=kpi,
                kpi_type='revenue',
                population=population,
                media=media if 'media' in locals() else None,
                media_spend=media_spend if 'media_spend' in locals() else None
            )
            logger.info("Successfully created InputData object")

            return input_data_obj

        except Exception as e:
            logger.error(f"Error creating InputData object: {str(e)}")
            raise ValueError(f"Failed to create InputData object: {str(e)}")

    except Exception as e:
        logger.error(f"Error in create_input_data: {str(e)}")
        raise ValueError(f"Failed to create InputData object: {str(e)}")

def sample_model(meridian_model: model.Meridian, request: AnalysisRequest) -> None:
    """Helper function to sample prior and posterior distributions."""
    # Get sampling parameters with defaults
    n_draws = request.n_draws or constants.DEFAULT_N_DRAWS
    n_chains = request.n_chains or 4
    n_adapt = request.n_adapt or 1000
    n_burnin = request.n_burnin or 1000
    n_keep = request.n_keep or 1000

    # Sample prior
    logger.info(f"Sampling prior distribution with {n_draws} draws...")
    meridian_model.sample_prior(n_draws=n_draws)

    # Sample posterior
    logger.info(f"Sampling posterior distribution with parameters:")
    logger.info(f"  n_chains: {n_chains}")
    logger.info(f"  n_adapt: {n_adapt}")
    logger.info(f"  n_burnin: {n_burnin}")
    logger.info(f"  n_keep: {n_keep}")

    meridian_model.sample_posterior(
        n_chains=n_chains,
        n_adapt=n_adapt,
        n_burnin=n_burnin,
        n_keep=n_keep
    )
    logger.info("Completed sampling")

@app.post("/analyze/predictive-accuracy")
async def analyze_predictive_accuracy(request: AnalysisRequest):
    """Analyze predictive accuracy of the model."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get predictive accuracy metrics
        result = analyzer_obj.predictive_accuracy(
            selected_geos=request.selected_geos,
            selected_times=request.selected_times
        )
        logger.info("Successfully performed predictive accuracy analysis")

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error in predictive accuracy analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/response-curves")
async def analyze_response_curves(request: AnalysisRequest):
    """Analyze response curves for media channels."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get response curves
        result = analyzer_obj.response_curves(
            confidence_level=request.confidence_level,
            selected_geos=request.selected_geos,
            selected_times=request.selected_times
        )
        logger.info("Successfully performed response curves analysis")

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error in response curves analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/hill-curves")
async def analyze_hill_curves(request: AnalysisRequest):
    """Analyze hill curves for media channels."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get hill curves
        result = analyzer_obj.hill_curves(confidence_level=request.confidence_level)
        logger.info("Successfully performed hill curves analysis")

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error in hill curves analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/adstock-decay")
async def analyze_adstock_decay(request: AnalysisRequest):
    """Analyze adstock decay for media channels."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get adstock decay analysis
        result = analyzer_obj.adstock_decay(confidence_level=request.confidence_level)
        logger.info("Successfully performed adstock decay analysis")

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error in adstock decay analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-data")
async def upload_data(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing uploaded file: {file.filename}")

        # Read the file in chunks
        chunk_size = 8192  # 8KB chunks
        contents = bytearray()

        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            contents.extend(chunk)

        # Convert to pandas DataFrame based on file type
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(contents), engine='c')  # Use C engine for faster parsing
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(BytesIO(contents), engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv or .xlsx file.")

        logger.info(f"File read successfully. Shape: {df.shape}")

        # Basic validation first
        required_columns = ['geo', 'date', 'kpi', 'population']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {', '.join(missing_columns)}. Required: {', '.join(required_columns)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Process data in chunks
        chunk_size = 10000  # Process 10,000 rows at a time
        total_rows = len(df)
        processed_df = pd.DataFrame()

        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx].copy()

            # Process chunk
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk['kpi'] = pd.to_numeric(chunk['kpi'], errors='coerce')
            chunk['population'] = pd.to_numeric(chunk['population'], errors='coerce')

            # Check for null values in this chunk
            null_counts = chunk[required_columns].isnull().sum()
            if null_counts.any():
                null_cols = [f"{col} ({count} nulls)" for col, count in null_counts.items() if count > 0]
                error_msg = f"Found null values in chunk {start_idx}-{end_idx}, columns: {', '.join(null_cols)}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            processed_df = pd.concat([processed_df, chunk])

        # Sort data (only after all chunks are processed)
        processed_df = processed_df.sort_values(['geo', 'date'])

        # Convert DataFrame to dictionary efficiently
        data_dict = {
            "columns": processed_df.columns.tolist(),
            "data": processed_df.values.tolist(),
            "index": list(range(len(processed_df)))
        }

        response_data = {
            "data": data_dict,
            "success": True,
            "message": f"Successfully processed {file.filename}",
            "summary": {
                "total_rows": len(processed_df),
                "unique_geos": processed_df['geo'].nunique(),
                "date_range": [processed_df['date'].min().strftime('%Y-%m-%d'), processed_df['date'].max().strftime('%Y-%m-%d')],
                "media_channels": [col.replace('_spend', '') for col in processed_df.columns if col.endswith('_spend')]
            }
        }

        logger.info(f"File processed successfully: {response_data['summary']}")
        return response_data

    except ValueError as ve:
        logger.error(f"Validation error processing file: {str(ve)}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/sample-data")
async def get_sample_data():
    """Return a sample data structure that shows the expected format."""
    # Read the sample marketing data file
    try:
        sample_file_path = '../sample_marketing_data.csv'
        df = pd.read_csv(sample_file_path)
        data_dict = df.to_dict(orient='split')

        return {
            "sample_data": {
                "columns": data_dict["columns"],
                "data": data_dict["data"],
                "index": list(range(len(data_dict["data"])))
            },
            "format_description": {
                "geo": "String: Geographic location identifier (e.g., 'NYC', 'LA')",
                "date": "Date string (YYYY-MM-DD): Time point",
                "kpi": "Float: Revenue/sales value (e.g., 150000)",
                "population": "Integer: Population size for the geographic location (e.g., 8400000)",
                "media_spend_columns": "Float: All media spend columns must end with '_spend' (e.g., 'tv_spend', 'facebook_spend')"
            },
            "dimension_requirements": {
                "media": ["geo", "time", "media_channel"],
                "media_spend": ["geo", "time", "media_channel"],
                "kpi": ["geo", "time"],
                "population": ["geo"]
            },
            "notes": [
                "All columns are required",
                "Dates should be in YYYY-MM-DD format",
                "KPI should be revenue/sales values",
                "Population values should be numeric and typically constant per geo",
                "All spend columns should be numeric and end with '_spend'",
                "Data should be sorted by geo and date",
                "Media and media_spend data will use [geo, time, media_channel] dimensions",
                "KPI data will use [geo, time] dimensions",
                "Population data will use [geo] dimension",
                "All time dimensions must align between KPI and media data"
            ]
        }
    except Exception as e:
        logger.error(f"Error reading sample data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error reading sample data: {str(e)}")

@app.post("/analyze/roi-analysis")
async def analyze_roi(request: AnalysisRequest):
    """Analyze ROI for media channels."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get ROI analysis
        result = analyzer_obj.roi(
            use_posterior=True,
            selected_geos=request.selected_geos,
            selected_times=request.selected_times,
            aggregate_geos=True,
            use_kpi=False
        )
        logger.info("Successfully performed ROI analysis")

        # Convert TensorFlow tensor to numpy array and then to dictionary
        result_np = result.numpy()

        # Calculate mean, confidence intervals
        confidence_level = request.confidence_level
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile

        # Get channel names
        channels = meridian_model.input_data.get_all_paid_channels()

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                "roi": (["channel", "metric"], [
                    [
                        float(np.mean(result_np, axis=(0, 1))[i]),  # mean
                        float(np.median(result_np, axis=(0, 1))[i]),  # median
                        float(np.percentile(result_np, lower_percentile * 100, axis=(0, 1))[i]),  # ci_lo
                        float(np.percentile(result_np, upper_percentile * 100, axis=(0, 1))[i])  # ci_hi
                    ]
                    for i in range(len(channels))
                ])
            },
            coords={
                "channel": channels,
                "metric": ["mean", "median", "ci_lo", "ci_hi"]
            },
            attrs={
                "confidence_level": confidence_level
            }
        )

        return ds.to_dict()

    except Exception as e:
        logger.error(f"Error in ROI analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/rhat-diagnostics")
async def analyze_rhat_diagnostics(request: AnalysisRequest):
    """Analyze Rhat diagnostics for model convergence."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get Rhat diagnostics
        result = analyzer_obj.get_rhat()
        logger.info("Successfully performed Rhat diagnostics")

        # Convert TensorFlow tensors to Python scalars
        result = {k: float(v.numpy()) if hasattr(v, 'numpy') else float(v) for k, v in result.items()}

        return result

    except Exception as e:
        logger.error(f"Error in Rhat diagnostics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/optimal-frequency")
async def analyze_optimal_frequency(request: AnalysisRequest):
    """Analyze optimal frequency for reach and frequency channels."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get optimal frequency analysis
        result = analyzer_obj.optimal_freq(
            use_posterior=True,
            use_kpi=False,
            selected_geos=request.selected_geos,
            selected_times=request.selected_times,
            confidence_level=request.confidence_level
        )
        logger.info("Successfully performed optimal frequency analysis")

        # Convert TensorFlow tensors to Python values in the xarray Dataset
        for var_name in result.data_vars:
            if hasattr(result[var_name].data, 'numpy'):
                result[var_name].data = result[var_name].data.numpy()

        return result.to_dict()

    except Exception as e:
        logger.error(f"Error in optimal frequency analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/cpik-analysis")
async def analyze_cpik(request: AnalysisRequest):
    """Analyze cost per incremental KPI (CPIK) for media channels."""
    try:
        # Create InputData object
        input_data_obj = create_input_data(request.data)
        logger.info("Successfully created InputData object")

        # Create and initialize Meridian model
        meridian_model = model.Meridian(input_data=input_data_obj)
        logger.info("Created Meridian model")

        # Sample prior and posterior distributions
        sample_model(meridian_model, request)

        # Create analyzer
        analyzer_obj = analysis.Analyzer(meridian=meridian_model)
        logger.info("Created analyzer object")

        # Get CPIK analysis
        result = analyzer_obj.cpik(
            use_posterior=True,
            selected_geos=request.selected_geos,
            selected_times=request.selected_times,
            aggregate_geos=True
        )
        logger.info("Successfully performed CPIK analysis")

        # Convert TensorFlow tensor to numpy array and then to dictionary
        result_np = result.numpy()

        # Calculate mean, confidence intervals
        confidence_level = request.confidence_level
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile

        # Get channel names
        channels = meridian_model.input_data.get_all_paid_channels()

        # Create xarray Dataset
        ds = xr.Dataset(
            data_vars={
                "cpik": (["channel", "metric"], [
                    [
                        float(np.mean(result_np, axis=(0, 1))[i]),  # mean
                        float(np.median(result_np, axis=(0, 1))[i]),  # median
                        float(np.percentile(result_np, lower_percentile * 100, axis=(0, 1))[i]),  # ci_lo
                        float(np.percentile(result_np, upper_percentile * 100, axis=(0, 1))[i])  # ci_hi
                    ]
                    for i in range(len(channels))
                ])
            },
            coords={
                "channel": channels,
                "metric": ["mean", "median", "ci_lo", "ci_hi"]
            },
            attrs={
                "confidence_level": confidence_level
            }
        )

        return ds.to_dict()

    except Exception as e:
        logger.error(f"Error in CPIK analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Entry point for local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
