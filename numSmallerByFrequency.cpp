class Solution {
public:
    vector<int> numByfrequency(vector<string>& q)
    {
        vector<int> res;
        for(vector<string>::iterator it=q.begin(); it < q.end(); ++it){
            char small_;
            int small_num=0;
            for(int j=0; j< (*it).size(); ++j){
                if(j==0 || (*it)[j]<small_){
                    small_num=0;
                    small_=(*it)[j];
                    ++small_num;
                }
                else if((*it)[j]==small_) ++small_num;
            }
            res.push_back(small_num);
        }
        return res;
    }
    vector<int> numSmallerByFrequency(vector<string>& queries, vector<string>& words) {
        vector<int> queries_num=numByfrequency(queries);
        vector<int> words_num=numByfrequency(words);
        vector<int> res;
        for(vector<int>::iterator iter=queries_num.begin(); iter<queries_num.end(); ++iter){
            int temp=0;
            for(vector<int>::iterator it=words_num.begin(); it<words_num.end(); ++it){
                if((*iter)<(*it)) ++temp;
            }
            res.push_back(temp);
        }
        return res;
    }
};