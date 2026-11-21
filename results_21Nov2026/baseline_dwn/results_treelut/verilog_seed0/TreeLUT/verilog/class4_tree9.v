module class4_tree9(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[429] & (~i[393])) | ((~i[67]) & (~i[71]) & i[393]) | (i[292] & (~i[429]) & (~i[393])) | ((~i[401]) & (~i[377]) & i[71] & i[393]) | ((~i[60]) & i[453] & i[377] & i[71] & i[393]) | ((~i[106]) & (~i[453]) & i[377] & i[71] & i[393]) | (i[229] & i[401] & (~i[377]) & i[71] & i[393]) | ((~i[297]) & (~i[126]) & i[67] & (~i[71]) & i[393]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
