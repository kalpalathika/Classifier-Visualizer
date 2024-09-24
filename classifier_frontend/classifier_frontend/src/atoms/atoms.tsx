import { atom } from 'recoil';

// Create an atom to hold the state value
export const predictResAtom = atom<any>({
  key: 'predictResAtom',
  default: {}
});

export const classifierTypeAtom = atom<string>({
  key: 'classifierTypeAtom',
  default: 'knn'
})