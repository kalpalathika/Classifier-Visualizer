export const apiDict : any = {
    "knn": {
        url:"http://localhost:5000/predict/knn",
        data: [
            {
                key:"n_neighbors",
                label: "Select n neighbors",
                type: "number",

            }
        ]
        
    },
    "svm": {
        url:"http://localhost:5000/predict/svm",

    }
}