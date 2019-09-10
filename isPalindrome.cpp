bool isPalindrome(int x){
    if(x<0) return false;
    deque<int> test;
    while(x){
    test.push_back(x%10);
    x/=10;
    }
    while(test.size()>1){
    if(test.front()==test.back()){
        test.pop_front();
        test.pop_back();
        continue;
    }
    return false;
    }
    return true;
}