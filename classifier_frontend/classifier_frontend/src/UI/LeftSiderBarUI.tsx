import { styled } from "styled-components";
import {Sidebar} from "./Siderbar"
import Dropdown from 'react-dropdown';
import { useRecoilState } from "recoil";
import { classifierTypeAtom } from "../atoms/atoms";

export const LeftSiderBarUI = () => {
    const [_,setClassifierType] = useRecoilState<any>(classifierTypeAtom);

    const classifierTypeList = [
        { value: 'knn', label: 'KNN' },
        { value: 'svm', label: 'SVM' },
        { value: 'bayes', label: 'Bayes' },
        { value: 'weighted-MAP', label: 'Weighted MAP' },



      ];
    const defaultOption = classifierTypeList[0].value;
    
    const onChange = (option: any) => {
        setClassifierType(option.value);
    }
    
    return(
    <Sidebar title={"Classifier Visualizer"}>
        <DropdownContainer>
            <DropdownTitle> Select Classifier Type</DropdownTitle>

            <Dropdown options={classifierTypeList} onChange={onChange} value={defaultOption} placeholder="Select an option" />
        </DropdownContainer>
        <DropdownContainer>
            <Dropdown options={classifierTypeList} onChange={onChange} value={defaultOption} placeholder="Select an option" />
        </DropdownContainer>
        <DropdownContainer>
            <Dropdown options={classifierTypeList} onChange={onChange} value={defaultOption} placeholder="Select an option" />
        </DropdownContainer>
    </Sidebar> 

    )



}


const DropdownContainer = styled.div`
  padding: 60px 20px; /* Adjust the padding value as needed */
`;

const DropdownTitle = styled.p`
  font-size: 14px;
  color: #e0ecf1;
  padding: 0px 10px;
  text-align: left;
`
