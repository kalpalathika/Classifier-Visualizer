import  {FC, ReactNode } from 'react';
import styled from 'styled-components';
import 'react-dropdown/style.css';
import SidebarHeader from './SidebarHeader';
interface BaseLayoutProps {
  title: String;
  children?: ReactNode;
}

export const Sidebar: FC<BaseLayoutProps> = ({title,children}) => {
  return (
    <SidebarContainer>
      <SidebarHeader title={title}/>
        {children}
    </SidebarContainer>
  );
};

const SidebarContainer = styled.div`
  height: 100vh  ;
  background-color: #202123;
  color: #202123;
`;

