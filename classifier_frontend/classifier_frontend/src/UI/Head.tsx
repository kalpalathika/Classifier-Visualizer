import  {FC } from 'react';
import styled from 'styled-components';

// Define the styles for the heading using the styled-components library

interface HeadProps {
  title: String;
}

const Heading = styled.div`
  font-size: 24px;
  font-weight: bold;
  color: #444654;
  padding: 30px 0px;  
`;

const Head: FC<HeadProps> = ({title}) => {
  return (
    <div>
      {/* Use the Heading component like a regular HTML heading */}
      <Heading>{title}</Heading>

    </div>
  );
};

export default Head;
