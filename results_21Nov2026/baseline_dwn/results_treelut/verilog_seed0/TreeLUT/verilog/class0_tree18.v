module class0_tree18(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[154] & (~i[219])) | ((~i[59]) & i[397] & i[219]) | ((~i[58]) & (~i[397]) & i[219]) | (i[375] & i[59] & i[397] & i[219]) | ((~i[375]) & i[59] & i[397] & i[219]) | (i[110] & i[107] & i[58] & (~i[397]) & i[219]) | ((~i[110]) & i[107] & i[58] & (~i[397]) & i[219]) | ((~i[251]) & (~i[107]) & i[58] & (~i[397]) & i[219]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
