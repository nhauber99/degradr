g++ -Wno-unused-result -Wsign-compare -DNDEBUG -g -fwrapv -O2 -Wall -g -fstack-protector-strong -Wformat -Werror=format-security -g -fwrapv -O2 -g -fstack-protector-strong -Wformat -Werror=format-security -Wdate-time -D_FORTIFY_SOURCE=2 -fPIC module.cpp -shared -o PyIPP.so -I ~/intel/oneapi/ipp/latest/include -I/usr/include/python3.10 -L ~/intel/oneapi/ipp/latest/lib/ -lippcore -lippvm -lipps -lippi -lippcc -std=c++20 -Bs