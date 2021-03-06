var express = require('express');
var app = express();
var fs = require('fs');
var df = require('dataformat');
 
mysql = require('mysql');
var connection = mysql.createConnection({
    host: 'localhost',
    user: 'monitor',
    password: 'abc123',
    database: 'activity'
})
connection.connect();
 
// server:3000/logone 에 GET 방식으로 접속하면 파라미터 값을 받아서 mysql에 넣는 작업을 수행하고
// log.txt 파일에 데이터를 저장하는 작업 또한 수행합니다
app.get('/logone', function(req, res){
    var i = 0;
    i++;
    
 
    r = {};
    r.seq = i;
    r.device = '102';
    r.ip = req.ip;
    r.value = req.query.state;
 
    var query = connection.query('insert into state set ?', r, function(err, rows,cols){
        if(err)
        {
            throw err;
        }
        console.log("[+]SQL injection is done!");
    });
    res.set('Content-Type','text/plain');
    res.send(200, r.state);

});

app.get('/s1', function(req, res){
    fs.readFile("/home/pi/Desktop/hot/s1.png", function (err, data) {
        res.writeHeader(200, {"Content-Type": "text/html"});
        res.write(data);
        res.end()
    });
});

app.use(express.static('pic'));
 
// localhost:3000/graph에 접속하면 googleChartSample.html에서 구글차트 템플릿을 읽어와 mysql의 데이터를 입력받아 뿌려줍니다
app.get('/graph', function(req, res){
    // 해당 템플릿 파일을 읽어온다
    var html = fs.readFile("/home/pi/Desktop/hot/monitor.html", function (err, html) {
        html = " "+ html;
 
        // mysql로부터 데이터를 읽어온다
        var qstr = 'select value from state order by time DESC';
        connection.query(qstr, function(err, rows, cols) {
            if (err) throw err;
 
            var data = rows[0].value;
            // sample.html의 <%HEADER%>와 <%DATA%> 부분을 데이터로 교체한다
            if (data == 0){
                html = html.replace("<%DATA%>", "정지");
                html = html.replace("<%IMAGE%>", "<img src=\"s2.png\" width=\"300\" height=\"300\" >");
            }else if (data == 1){
                html = html.replace("<%DATA%>", "걷기");
                html = html.replace("<%IMAGE%>", "<img src=\"w1.png\" width=\"300\" height=\"300\" >");
            }else if (data == 2){
                html = html.replace("<%DATA%>", "앉기/일어서기");
                html = html.replace("<%IMAGE%>", "<img src=\"s1.png\" width=\"300\" height=\"300\" >");
            }else if (data == 3){
                html = html.replace("<%DATA%>", "뛰기");
                html = html.replace("<%IMAGE%>", "<img src=\"r1.png\" width=\"300\" height=\"300\" >");
            }else if (data == 4){
                html = html.replace("<%DATA%>", "낙상");
                html = html.replace("<%IMAGE%>", "<img src=\"f1.png\" width=\"300\" height=\"300\" >");
            }
        });
        qstr = 'select count(*) as count from state group by value having value>=0';
        connection.query(qstr, function(err, rows, cols) {
            if (err) throw err;
            
            html = html.replace("<%DATASTP%>", rows[0].count);
            html = html.replace("<%DATAWLK%>", rows[1].count);
            html = html.replace("<%DATARUN%>", rows[2].count);
            html = html.replace("<%DATASIT%>", rows[3].count);
            html = html.replace("<%DATAFAL%>", rows[4].count);
            res.writeHeader(200, {"Content-Type": "text/html"});
            res.write(html);
            res.end();
        });
    });
});


// 3000번 포트를 사용합니다
var server = app.listen(3000, function(){
    var host = server.address().address
    var port = server.address().port
    console.log('listening at http://%s:%s',host,port)
});

