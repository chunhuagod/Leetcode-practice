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

