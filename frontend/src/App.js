import React, { useState } from 'react';
import {
  AppBar,
  Box,
  Button,
  Container,
  CssBaseline,
  Paper,
  ThemeProvider,
  Toolbar,
  Typography,
  createTheme,
  CircularProgress,
  Alert,
  Snackbar,
  Grid,
  Slider,
  Divider,
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';
import api from './services/api';

// Only import ROI analysis chart
import RoiAnalysisChart from './components/RoiAnalysisChart';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function App() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [confidenceLevel, setConfidenceLevel] = useState(0.9);
  const [results, setResults] = useState(null);
  const [analysisType, setAnalysisType] = useState(null);

  const handleFileUpload = async (event) => {
    if (event.target.files && event.target.files[0]) {
      const uploadedFile = event.target.files[0];
      setFile(uploadedFile);
      setLoading(true);
      setError(null);

      try {
        const response = await api.uploadData(uploadedFile);
        if (response.success) {
          setData(response.data);
          console.log('Data loaded successfully:', response.data);
        } else {
          throw new Error('Failed to process file');
        }
      } catch (err) {
        console.error('Upload error:', err);
        setError(err.response?.data?.detail || err.message || 'Failed to upload file');
        setData(null);
      } finally {
        setLoading(false);
      }
    }
  };

  const handleAnalysis = async (type) => {
    if (!data) {
      setError('Please upload data first');
      return;
    }

    setLoading(true);
    setError(null);
    setAnalysisType(type);

    try {
      let response;
      const request = {
        data: data,
        confidence_level: confidenceLevel,
      };

      console.log('Sending analysis request:', request);

      switch (type) {
        case 'predictive-accuracy':
          response = await api.getPredictiveAccuracy(request);
          break;
        case 'response-curves':
          response = await api.getResponseCurves(request);
          break;
        case 'hill-curves':
          response = await api.getHillCurves(request);
          break;
        case 'adstock-decay':
          response = await api.getAdstockDecay(request);
          break;
        case 'roi':
          response = await api.getRoi(request);
          break;
        case 'rhat':
          response = await api.getRhat(request);
          break;
        case 'optimal-frequency':
          response = await api.getOptimalFrequency(request);
          break;
        case 'cpik':
          response = await api.getCpik(request);
          break;
        default:
          throw new Error('Invalid analysis type');
      }

      setResults(response);
    } catch (err) {
      console.error('Analysis error:', err);
      setError(err.response?.data?.detail || err.message || `Failed to perform ${type} analysis`);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const renderAnalysisResults = () => {
    if (!results) return null;

    // Only show chart for ROI analysis
    if (analysisType === 'roi') {
      return <RoiAnalysisChart data={results} />;
    }

    // For all other analyses, show the raw response data in a formatted way
    return (
      <Paper sx={{ p: 3, mt: 3 }}>
        <Typography variant="h6" gutterBottom>
          {analysisType.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')} Results
        </Typography>
        <Box
          component="pre"
          sx={{
            mt: 2,
            p: 2,
            bgcolor: '#f5f5f5',
            borderRadius: 1,
            overflow: 'auto',
            maxHeight: '500px'
          }}
        >
          {JSON.stringify(results, null, 2)}
        </Box>
      </Paper>
    );
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ flexGrow: 1 }}>
        <AppBar position="static">
          <Toolbar>
            <Typography variant="h6" sx={{ flexGrow: 1 }}>
              Meridian Analysis Tool
            </Typography>
          </Toolbar>
        </AppBar>
        <Container maxWidth="lg" sx={{ mt: 4 }}>
          <Paper sx={{ p: 4 }}>
            <Typography variant="h4" gutterBottom>
              Welcome to Meridian
            </Typography>
            <Typography variant="body1" paragraph>
              Upload your data and perform advanced analysis using the Meridian package.
            </Typography>

            <Grid container spacing={4}>
              <Grid item xs={12}>
                <Button
                  variant="contained"
                  component="label"
                  startIcon={<CloudUpload />}
                  sx={{ mb: 2 }}
                >
                  Upload Data
                  <input
                    type="file"
                    hidden
                    accept=".csv,.xlsx"
                    onChange={handleFileUpload}
                  />
                </Button>
                {file && (
                  <Typography variant="body2" sx={{ ml: 2 }}>
                    Selected file: {file.name}
                  </Typography>
                )}
              </Grid>

              <Grid item xs={12}>
                <Typography gutterBottom>
                  Confidence Level: {confidenceLevel}
                </Typography>
                <Slider
                  value={confidenceLevel}
                  onChange={(_, value) => setConfidenceLevel(value)}
                  min={0.1}
                  max={0.99}
                  step={0.01}
                  sx={{ width: 300 }}
                />
              </Grid>

              <Grid item xs={12}>
                <Typography variant="h6" gutterBottom>
                  Core Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item>
                    <Button
                      variant="contained"
                      onClick={() => handleAnalysis('predictive-accuracy')}
                      disabled={!data || loading}
                    >
                      Predictive Accuracy
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      onClick={() => handleAnalysis('response-curves')}
                      disabled={!data || loading}
                    >
                      Response Curves
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      onClick={() => handleAnalysis('hill-curves')}
                      disabled={!data || loading}
                    >
                      Hill Curves
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      onClick={() => handleAnalysis('adstock-decay')}
                      disabled={!data || loading}
                    >
                      Adstock Decay
                    </Button>
                  </Grid>
                </Grid>
              </Grid>

              <Grid item xs={12}>
                <Divider sx={{ my: 2 }} />
                <Typography variant="h6" gutterBottom>
                  Advanced Analysis
                </Typography>
                <Grid container spacing={2}>
                  <Grid item>
                    <Button
                      variant="contained"
                      color="secondary"
                      onClick={() => handleAnalysis('roi')}
                      disabled={!data || loading}
                    >
                      ROI Analysis
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      color="secondary"
                      onClick={() => handleAnalysis('rhat')}
                      disabled={!data || loading}
                    >
                      R-hat Diagnostics
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      color="secondary"
                      onClick={() => handleAnalysis('optimal-frequency')}
                      disabled={!data || loading}
                    >
                      Optimal Frequency
                    </Button>
                  </Grid>
                  <Grid item>
                    <Button
                      variant="contained"
                      color="secondary"
                      onClick={() => handleAnalysis('cpik')}
                      disabled={!data || loading}
                    >
                      CPIK Analysis
                    </Button>
                  </Grid>
                </Grid>
              </Grid>

              {loading && (
                <Grid item xs={12}>
                  <CircularProgress />
                </Grid>
              )}

              {renderAnalysisResults()}
            </Grid>
          </Paper>
        </Container>
      </Box>

      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={() => setError(null)}
      >
        <Alert onClose={() => setError(null)} severity="error">
          {error}
        </Alert>
      </Snackbar>
    </ThemeProvider>
  );
}

export default App;
