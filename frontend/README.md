# Meridian Web Interface

This is a web interface for the Meridian package that allows users to interact with the package's functionality through a modern web application.

## Prerequisites

- Node.js (v14 or later)
- Python (v3.7 or later)
- pip (Python package manager)

## Project Structure

```
meridian/
├── frontend/    # React frontend application
└── backend/     # FastAPI backend server
```

## Setup and Installation

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```

The backend will be available at http://localhost:8000

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The frontend will be available at http://localhost:3000

## Usage

1. Open your web browser and navigate to http://localhost:3000
2. Use the interface to interact with Meridian's functionality
3. The results will be displayed in the web interface

## Development

- Frontend code is in `frontend/src/`
- Backend code is in `backend/`
- Add new API endpoints in `backend/main.py`
- Add new frontend components in `frontend/src/components/`

## Contributing

Please read the CONTRIBUTING.md file in the root directory for details on contributing to this project.

## License

This project is licensed under the same terms as the Meridian package. See the LICENSE file in the root directory for details.
