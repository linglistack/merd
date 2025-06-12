import axios from 'axios';

// Use environment variable for API URL, fallback to localhost if not set
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const handleError = (error) => {
  if (error.response) {
    // The request was made and the server responded with a status code
    // that falls out of the range of 2xx
    throw error;
  } else if (error.request) {
    // The request was made but no response was received
    throw new Error('No response from server. Please check if the server is running.');
  } else {
    // Something happened in setting up the request that triggered an Error
    throw new Error('Error setting up the request: ' + error.message);
  }
};

const api = {
  uploadData: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post(`${API_BASE_URL}/upload-data`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getPredictiveAccuracy: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/predictive-accuracy`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getResponseCurves: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/response-curves`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getHillCurves: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/hill-curves`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getAdstockDecay: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/adstock-decay`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getRoi: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/roi-analysis`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getRhat: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/rhat-diagnostics`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getOptimalFrequency: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/optimal-frequency`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },

  getCpik: async (request) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/analyze/cpik-analysis`, request);
      return response.data;
    } catch (error) {
      handleError(error);
    }
  },
};

export default api;
