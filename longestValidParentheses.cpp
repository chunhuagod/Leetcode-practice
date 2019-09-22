int longestValidParentheses(string s) {
    vector<int> vec;
    int length=0,index=0;
    for(int i=0;i<s.size();++i){
        if(s[i]=='(') vec.push_back(i);
        else{
            if(vec.size()==0) index=i+1;
            else {vec.pop_back();
                length=(vec.size())? max(length,i-vec.back()):max(length,i-index+1);}
        }
    }
    return length;
}