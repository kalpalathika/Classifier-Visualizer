import { Sidebar } from "./Siderbar"
import {useRecoilValue } from 'recoil';
import { predictResAtom } from "../atoms/atoms";
import { styled } from "styled-components";
import ConfusionMatrix from "./ConfusionMatrix";

export const RightSiderBar = () => {
    const response = useRecoilValue<any>(predictResAtom);
    return <Sidebar title="Results">
        <TextContainter>
            <Text>Accuracy: {response.recognition_rate} %</Text>
            { response.confusion_matrix ? <ConfusionMatrix confusions={response.confusion_matrix}/> : null}
        </TextContainter>
    </Sidebar>
}

const TextContainter = styled.div`
    display: flex;
    justify-content: space-between;
    flex-direction: column;
    align-items: center;
    margin-top: 100px;

`
const Text = styled.h3`
    color: #fff;
`