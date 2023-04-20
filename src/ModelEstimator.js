import React, {useEffect, useState} from 'react';
import Plot from 'react-plotly.js';

import {calculateFlops, calculateLossData, calculateTrainingCost, calculateTrainingTime} from './FlopsCalculator';


export const ModelEstimator = ({params}) => {
    const [plotData, setPlotData] = useState(null);

    useEffect(() => {
        // Check if any of the parameters are 0
        const hasZeroParams = Object.values(params).some((param) => param === 0);

        // Only set plotData if all parameters are non-zero
        if (!hasZeroParams) {
            const {S_values, losses} = calculateLossData(params);
            const data = [
                {
                    x: S_values,
                    y: losses,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Loss vs Training Steps',
                },
            ];
            setPlotData(data);
        }
    }, [params]);

    // Perform calculations and update the model training estimations
    const flops = calculateFlops(params);
    const trainingTime = calculateTrainingTime(flops, params);
    const trainingCost = calculateTrainingCost(trainingTime, params);

    return (
        <div>
            <h3>Model Training Estimations:</h3>
            <p>Required Flop: {flops ? flops.toFixed(2) : 0} T</p>
            <p>Training Time: {trainingTime ? trainingTime.toFixed(2) : 0} hours</p>
            <p>Training Cost: ${trainingCost ? trainingCost.toFixed(2) : 0}</p>
            {plotData && (
                <div>
                    <h3>Loss and Training Steps Relationship:</h3>
                    <Plot
                        data={plotData}
                        layout={{
                            width: 500,
                            height: 300,
                            title: 'Loss vs Training Steps',
                            xaxis_title: 'Training Steps',
                            yaxis_title: 'Loss',
                        }}
                    />
                </div>
            )}
        </div>
    );
};