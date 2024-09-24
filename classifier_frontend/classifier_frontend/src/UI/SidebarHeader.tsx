import {FC } from 'react';
import styled from 'styled-components';

interface SidebarHeaderProps {
  title: String;
}

// Create a styled component for the fixed title in the Sidebar
const SidebarTitle = styled.div`
  width: 100%;
  background-color: #202123;
  color: #00FF00;
  font-size: 18px;
  font-weight: bold;
  box-shadow: 6px 5px 10px rgba(0, 0, 0, 5);
  padding: 20px 0px;
`;

const SidebarHeader: FC<SidebarHeaderProps> =  ({title}) => {
  return (
    <>
      <SidebarTitle>{title}</SidebarTitle>
    </>
  );
};

export default SidebarHeader;
