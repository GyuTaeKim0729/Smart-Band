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
app.get('/logone', function (req, res) {
    var i = 0;
    i++;

    r = {};
    r.seq = i;
    r.device = '102';
    r.ip = req.ip;
    r.value = req.query.state;

    var query = connection.query('insert into state set ?', r, function (err, rows, cols) {
        if (err) {
            throw err;
        }
        console.log("[+]SQL injection is done!");
        // 여기 어제 코드 넣어주라
    });
});


// localhost:3000/graph에 접속하면 googleChartSample.html에서 구글차트 템플릿을 읽어와 mysql의 데이터를 입력받아 뿌려줍니다
app.get('/graph', function (req, res) {
    // 해당 템플릿 파일을 읽어온다
    var html = fs.readFile("/home/pi/gitrepo/lockdpwn/systemNewTechnology/mission6/googleChartSample.html", function (err, html) {
        html = " " + html

        // mysql로부터 데이터를 읽어온다
        var qstr = 'select * from state';
        connection.query(qstr, function (err, rows, cols) {
            if (err) throw err;

            var data = "data.addRows([";
            var comma = "";

            // 데이터를 날짜/숫자 형식에 맞게 넣어준다
            for (var i = 0; i < rows.length; i++) {
                r = rows[i];
                data += comma + "[new Date(" + r.time.getFullYear() + "," + r.time.getMonth() + "," + r.time.getDate() + "," + r.time.getHours() + "," + r.time.getMinutes() + "," + r.time.getSeconds() + ")" + "," + r.value + "]";
                comma = ",";
            }

            // google chart 형식에 맞게 데이터를 설정한다
            var header = "data.addColumn('date', 'Date/Time');"
            header += "data.addColumn('number', 'activity');"
            data += "]);";

            // sample.html의 <%HEADER%>와 <%DATA%> 부분을 데이터로 교체한다
            html = html.replace("<% HEADER %>", header);
            html = html.replace("<% DATA %>", data);
            html = html.replace("<% STATE %>", rows[rows.length-1]);


            res.writeHeader(200, { "Content-Type": "text/html" });
            res.write(html);
            res.end();
        });
    });
})

// 3000번 포트를 사용합니다
var server = app.listen(3000, function () {
    var host = server.address().address
    var port = server.address().port
    console.log('listening at http://%s:%s', host, port)
});

