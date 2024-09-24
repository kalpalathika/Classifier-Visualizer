import React, { useEffect, useState } from 'react';
import { styled } from 'styled-components';
import loaderSvg from '../assets/loader.svg';
import { predictResAtom } from '../atoms/atoms';
import { useRecoilState } from 'recoil';

interface ClassifierFuncProps {
  apiUrl: string;
}

const ClassifierFunc: React.FC<ClassifierFuncProps> = ({ apiUrl }) => {
  const [response, setResponse] = useRecoilState<any>(predictResAtom);
  const [selectedNumber, setSelectedNumber] = useState<number | undefined>(undefined);
  const [fetchImage, setFetchImage] = useState(false);

  useEffect(() => {
    let debounceTimeout: ReturnType<typeof setTimeout> | null = null;

    if (fetchImage && selectedNumber !== undefined) {
      // Clear any existing timeout
      if (debounceTimeout !== null) {
        clearTimeout(debounceTimeout);
      }

      // Set a new timeout to make the API call after the user has stopped interacting
      debounceTimeout = setTimeout(async () => {
        try {
          const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              type: 'knn',
              knn_number: selectedNumber,
            }),
          });

          const data = await response.json();
          const prediction_response: any = data.response;
          setResponse(prediction_response);
          setFetchImage(false);
        } catch (error) {
          console.error('Error fetching image:', error);
          setFetchImage(false);
        }
      }, 3000); // Adjust the debounce delay as needed
    }

    // Cleanup function to clear the timeout when the component unmounts or the dependencies change
    return () => {
      if (debounceTimeout !== null) {
        clearTimeout(debounceTimeout);
      }
    };
  }, [fetchImage, selectedNumber, apiUrl]);

  const handleNumberChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newNumber = parseInt(event.target.value, 10);
    setSelectedNumber(newNumber);
    setFetchImage(true); // Trigger fetching when the number changes
    
  };



  return (
    <div>
      <label htmlFor="numberInput">Select a number:</label>
      <input
        type="number"
        id="numberInput"
        onChange={handleNumberChange}
        value={selectedNumber !== undefined ? selectedNumber : ''}
      />
      {response.prediction_blob ? (
        <ImageContainer>
          <img src={`data:image/png;base64,${response.prediction_blob}`} alt="My Image" />
        </ImageContainer>
      ) : (
        <ImageContainer>
          <img src={loaderSvg} alt="Loading..." />
        </ImageContainer>
      )}
    </div>
  );
};

export default ClassifierFunc;

const ImageContainer = styled.div`
  padding: 30px 0px;
`;
