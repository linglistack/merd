import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ErrorBar } from 'recharts';
import { Box, Typography, Paper } from '@mui/material';

const RoiAnalysisChart = ({ data }) => {
  if (!data || !data.coords || !data.data_vars) {
    return <Typography>No data available</Typography>;
  }

  // Extract channels
  const channels = data.coords.channel.data;

  // Transform data for Recharts
  const chartData = channels.map((channel, index) => ({
    channel,
    roi: data.data_vars.roi.data[index][0], // mean
    ci_lo: data.data_vars.roi.data[index][1],
    ci_hi: data.data_vars.roi.data[index][2],
  }));

  return (
    <Paper elevation={3} sx={{ p: 3, m: 2 }}>
      <Typography variant="h6" gutterBottom>
        ROI Analysis by Channel
      </Typography>
      <Box sx={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="channel"
              label={{ value: 'Channel', position: 'bottom' }}
            />
            <YAxis
              label={{ value: 'Return on Investment (ROI)', angle: -90, position: 'left' }}
            />
            <Tooltip />
            <Legend />
            <Bar
              dataKey="roi"
              fill="#1f77b4"
              name="ROI"
            >
              <ErrorBar
                dataKey={[["ci_lo"], ["ci_hi"]]}
                width={4}
                strokeWidth={2}
                stroke="#2c3e50"
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Box>
    </Paper>
  );
};

export default RoiAnalysisChart;
