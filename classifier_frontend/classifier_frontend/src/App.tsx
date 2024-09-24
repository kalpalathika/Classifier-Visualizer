import { styled } from 'styled-components'
import './App.css'
import ClassifierFunc from './classifier_components/ClassifierFunc'
import Head from './UI/Head'
import { LeftSiderBarUI } from './UI/LeftSiderBarUI'
import { RightSiderBar } from './UI/RightSiderBar'
import { useRecoilValue } from 'recoil';
import { classifierTypeAtom } from './atoms/atoms'
import { apiDict } from './constants/apiObj'


function App() {
  const classifierType : string = useRecoilValue<any>(classifierTypeAtom);


  return (
      <LayoutContainer>
        <SidebarContainer>
          <LeftSiderBarUI/>
        </SidebarContainer>
        <HeadContainer>
          <Head title={"abc"}/>
          <ClassifierFunc apiUrl={apiDict[classifierType].url}/>
        </HeadContainer>
        <SidebarContainer>
          <RightSiderBar/>
        </SidebarContainer>
      </LayoutContainer>
  )
}

export default App;

const LayoutContainer = styled.div`
display: flex;
`;
const HeadContainer = styled.div`
flex: 50%;
background: #444654;
`;

const SidebarContainer = styled.div`
flex: 20%;
`;

