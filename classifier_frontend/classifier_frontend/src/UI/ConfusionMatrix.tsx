import React from 'react';
import { styled } from 'styled-components';

interface ConfusionMatrixProps {
  confusions: number[][];
}


const ConfusionMatrix: React.FC<ConfusionMatrixProps> = ({ confusions }) => {
 console.log("confusion matrix---",confusions)
  return (
    <ConfusionMatrixContainer>
      <h3>Confusion Matrix:</h3>
      <table>
        <thead>
          <tr>
            <th>{' '}</th>
            <th>0: Blue-True</th>
            <th>{' '}</th>
            <th>1: Org-True</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th>0: Blue-Pred</th>
            <td>{confusions[0][0]}</td>
            <th>{''}</th>
            <td>{confusions[0][1]}</td>
          </tr>
          <tr>
            <th>1: Org-Pred</th>
            {/* <th>{' '}</th> */}
            <td>{confusions[1][0]}</td>
            <th>{' '}</th>  
            <td>{confusions[1][1]}</td>
          </tr>
        </tbody>
      </table>
    </ConfusionMatrixContainer>
  );
};

export default ConfusionMatrix;

const ConfusionMatrixContainer = styled.div`
    color: #fff;
`;