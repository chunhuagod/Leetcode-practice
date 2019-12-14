# Leetcode-practice

## LRU缓存

### 使用哈希表和双向链表实现 利用哈希表查找复杂度和链表的添加删除复杂度

```c++
class LRUCache {
private:
    int capacity;
    list<pair<int,int>> cache;
     unordered_map<int, list<pair<int, int>>::iterator> map;    
public:
    LRUCache(int capacity) {
        this->capacity=capacity;
    }
    
    int get(int key) {
        auto it=map.find(key);
        if(it == map.end()) return -1;
        pair<int,int> ans= *map[key];
        cache.erase(map[key]);
        cache.push_front(ans);
        map[key]= cache.begin();
        return ans.second;
    }
    
    void put(int key, int value) {
        auto it=map.find(key);
        if(it == map.end()){
            if(map.size()==capacity){
                auto temp=cache.back();
                map.erase(temp.first);
                cache.pop_back();
            }
            cache.push_front(make_pair(key,value));
            map[key]=cache.begin();
            return;
        }
        cache.erase(map[key]);
        cache.push_front(make_pair(key,value));
        map[key]= cache.begin();
    }
};
```

## 三数相加为0

```c++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int N = nums.size();
        vector<vector<int> > res;
        for (int i = 0; i < N - 2; ++i) {
            if (nums[i] > 0) break;
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            int l = i + 1;
            int r = N - 1;
            while (l < r) {
                int s = nums[i] + nums[l] + nums[r];
                if (s > 0) {
                    --r;
                } else if (s < 0) {
                    ++l;
                } else {
                    res.push_back({nums[i], nums[l], nums[r]});
                    while (l < r && nums[l] == nums[++l]);
                    while (l < r && nums[r] == nums[--r]);
                }
            }
        }
        return res;
    }
};
```



## String的乘法

### 基于竖乘原则来进行编程 降低复杂度

```c++
class Solution {
public:
    string multiply(string num1, string num2) {
        int length1=num1.size(),length2=num2.size();
        string res(length1+length2,'0');

        for(int i=length1-1; i>=0; --i){
            for (int j=length2-1; j>=0; --j){
                int temp=res[i+j+1]-'0'+(num1[i]-'0')*(num2[j]-'0');
                res[i+j+1]='0'+temp%10;
                res[i+j]+=temp/10;
            }
        }

        for(int i=0; i<res.size();++i){
            if (res[i]-'0') return res.substr(i);
        }

        return "0";
    }
};
```

